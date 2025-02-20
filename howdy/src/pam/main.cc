#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <ostream>

#include <glob.h>
#include <libintl.h>
#include <pthread.h>
#include <spawn.h>
#include <stdexcept>
#include <sys/signalfd.h>
#include <sys/syslog.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <syslog.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <vector>

#include <INIReader.h>

#include <security/pam_appl.h>
#include <security/pam_ext.h>
#include <security/pam_modules.h>

#include "enter_device.hh"
#include "main.hh"
#include "optional_task.hh"

const auto DEFAULT_TIMEOUT =
    std::chrono::duration<int, std::chrono::milliseconds::period>(100);
const auto MAX_RETRIES = 5;
const auto PYTHON_EXECUTABLE = "python3";
const auto COMPARE_PROCESS_PATH = "/lib64/security/howdy/compare.py";

#define S(msg) gettext(msg)

/**
 * Inspect the status code returned by the compare process
 * @param  status        The status code
 * @param  conv_function The PAM conversation function
 * @return               A PAM return code
 */
auto howdy_error(int status,
                 const std::function<int(int, const char *)> &conv_function)
    -> int
{
  // If the process has exited
  if (WIFEXITED(status))
  {
    // Get the status code returned
    status = WEXITSTATUS(status);

    switch (status)
    {
    case CompareError::NO_FACE_MODEL:
      conv_function(PAM_ERROR_MSG, S("There is no face model known"));
      syslog(LOG_NOTICE, "Failure, no face model known");
      break;
    case CompareError::TIMEOUT_REACHED:
      syslog(LOG_ERR, "Failure, timeout reached");
      break;
    case CompareError::ABORT:
      syslog(LOG_ERR, "Failure, general abort");
      break;
    case CompareError::TOO_DARK:
      conv_function(PAM_ERROR_MSG, S("Face detection image too dark"));
      syslog(LOG_ERR, "Failure, image too dark");
      break;
    default:
      conv_function(PAM_ERROR_MSG,
                    std::string(S("Unknown error: ") + status).c_str());
      syslog(LOG_ERR, "Failure, unknown error %d", status);
    }
  }
  else if (WIFSIGNALED(status))
  {
    // We get the signal
    status = WTERMSIG(status);

    syslog(LOG_ERR, "Child killed by signal %s (%d)", strsignal(status),
           status);
  }

  // As this function is only called for error status codes, signal an error to
  // PAM
  return PAM_AUTH_ERR;
}

/**
 * Format the success message if the status is successful or log the error in
 * the other case
 * @param  username      Username
 * @param  status        Status code
 * @param  config        INI  configuration
 * @param  conv_function PAM conversation function
 * @return          Returns the conversation function return code
 */
auto howdy_status(char *username, int status, const INIReader &config,
                  const std::function<int(int, const char *)> &conv_function)
    -> int
{
  if (status != EXIT_SUCCESS)
  {
    return howdy_error(status, conv_function);
  }

  if (!config.GetBoolean("core", "no_confirmation", true))
  {
    // Construct confirmation text from i18n string
    std::string confirm_text(S("Identified face as {}"));
    std::string identify_msg =
        confirm_text.replace(confirm_text.find("{}"), 2, std::string(username));
    conv_function(PAM_TEXT_INFO, identify_msg.c_str());
  }

  syslog(LOG_INFO, "Login approved");

  return PAM_SUCCESS;
}

/**
 * Check if Howdy should be enabled according to the configuration and the
 * environment.
 * @param  config INI configuration
 * @return        Returns PAM_AUTHINFO_UNAVAIL if it shouldn't be enabled,
 * PAM_SUCCESS otherwise
 */
auto check_enabled(const INIReader &config) -> int
{
  // Stop executing if Howdy has been disabled in the config
  if (config.GetBoolean("core", "disabled", false))
  {
    syslog(LOG_INFO, "Skipped authentication, Howdy is disabled");
    return PAM_AUTHINFO_UNAVAIL;
  }

  // Stop if we're in a remote shell and configured to exit
  if (config.GetBoolean("core", "ignore_ssh", true))
  {
    if (getenv("SSH_CONNECTION") != nullptr ||
        getenv("SSH_CLIENT") != nullptr || getenv("SSHD_OPTS") != nullptr)
    {
      syslog(LOG_INFO, "Skipped authentication, SSH session detected");
      return PAM_AUTHINFO_UNAVAIL;
    }
  }

  // Try to detect the laptop lid state and stop if it's closed
  if (config.GetBoolean("core", "ignore_closed_lid", true))
  {
    glob_t glob_result;

    // Get any files containing lid state
    int return_value =
        glob("/proc/acpi/button/lid/*/state", 0, nullptr, &glob_result);

    if (return_value != 0)
    {
      syslog(LOG_ERR, "Failed to read files from glob: %d", return_value);
      if (errno != 0)
      {
        syslog(LOG_ERR, "Underlying error: %s (%d)", strerror(errno), errno);
      }
    }
    else
    {
      for (size_t i = 0; i < glob_result.gl_pathc; i++)
      {
        std::ifstream file(std::string(glob_result.gl_pathv[i]));
        std::string lid_state;
        std::getline(file, lid_state, static_cast<char>(file.eof()));

        if (lid_state.find("closed") != std::string::npos)
        {
          globfree(&glob_result);

          syslog(LOG_INFO, "Skipped authentication, closed lid detected");
          return PAM_AUTHINFO_UNAVAIL;
        }
      }
    }
    globfree(&glob_result);
  }

  return PAM_SUCCESS;
}

/**
 * The main function, runs the identification and authentication
 * @param  pamh     The handle to interface directly with PAM
 * @param  flags    Flags passed on to us by PAM, XORed
 * @param  argc     Amount of rules in the PAM config (disregared)
 * @param  argv     Options defined in the PAM config
 * @param  auth_tok True if we should ask for a password too
 * @return          Returns a PAM return code
 */
auto identify(pam_handle_t *pamh, int flags, int argc, const char **argv,
              bool auth_tok) -> int
{
  INIReader config("/lib64/security/howdy/config.ini");
  openlog("pam_howdy", 0, LOG_AUTHPRIV);

  // Error out if we could not read the config file
  if (config.ParseError() != 0)
  {
    syslog(LOG_ERR, "Failed to parse the configuration file: %d",
           config.ParseError());
    return PAM_SYSTEM_ERR;
  }

  // Will contain the responses from PAM functions
  int pam_res = PAM_IGNORE;

  // Check if we shoud continue
  if ((pam_res = check_enabled(config)) != PAM_SUCCESS)
  {
    return pam_res;
  }

  // Will contain PAM conversation structure
  struct pam_conv *conv = nullptr;
  const void **conv_ptr =
      const_cast<const void **>(reinterpret_cast<void **>(&conv));

  if ((pam_res = pam_get_item(pamh, PAM_CONV, conv_ptr)) != PAM_SUCCESS)
  {
    syslog(LOG_ERR, "Failed to acquire conversation");
    return pam_res;
  }

  // Wrap the PAM conversation function in our own, easier function
  auto conv_function = [conv](int msg_type, const char *msg_str)
  {
    const struct pam_message msg = {.msg_style = msg_type, .msg = msg_str};
    const struct pam_message *msgp = &msg;

    struct pam_response res = {};
    struct pam_response *resp = &res;

    return conv->conv(1, &msgp, &resp, conv->appdata_ptr);
  };

  // Initialize gettext
  setlocale(LC_ALL, "");
  bindtextdomain(GETTEXT_PACKAGE, LOCALEDIR);
  textdomain(GETTEXT_PACKAGE);

  // If enabled, send a notice to the user that facial login is being attempted
  if (config.GetBoolean("core", "detection_notice", false))
  {
    if ((conv_function(PAM_TEXT_INFO, S("Attempting facial authentication"))) !=
        PAM_SUCCESS)
    {
      syslog(LOG_ERR, "Failed to send detection notice");
    }
  }

  // Get the username from PAM, needed to match correct face model
  char *username = nullptr;
  if ((pam_res = pam_get_user(pamh, const_cast<const char **>(&username),
                              nullptr)) != PAM_SUCCESS)
  {
    syslog(LOG_ERR, "Failed to get username");
    return pam_res;
  }

  // const char *const args[] = {PYTHON_EXECUTABLE, // NOLINT
  //                             COMPARE_PROCESS_PATH, username, nullptr};
  const char *const args[] = {"/lib64/security/howdy/howdy-auth",
                              username, nullptr};
  pid_t child_pid;

  // Start the python subprocess
  // if (posix_spawnp(&child_pid, PYTHON_EXECUTABLE, nullptr, nullptr,
  //                  const_cast<char *const *>(args), nullptr) != 0) {
  if (posix_spawnp(&child_pid, "/lib64/security/howdy/howdy-auth", nullptr, nullptr,
                   const_cast<char *const *>(args), nullptr) != 0)
  {
    syslog(LOG_ERR, "Can't spawn the howdy process: %s (%d)", strerror(errno),
           errno);
    return PAM_SYSTEM_ERR;
  }

  int status;
  waitpid(child_pid, &status, 0);

  return howdy_status(username, status, config, conv_function);
}

// Called by PAM when a user needs to be authenticated, for example by running
// the sudo command
PAM_EXTERN auto pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc,
                                    const char **argv) -> int
{
  return identify(pamh, flags, argc, argv, false);
}

// Called by PAM when a session is started, such as by the su command
PAM_EXTERN auto pam_sm_open_session(pam_handle_t *pamh, int flags, int argc,
                                    const char **argv) -> int
{
  return identify(pamh, flags, argc, argv, false);
}

// The functions below are required by PAM, but not needed in this module
PAM_EXTERN auto pam_sm_acct_mgmt(pam_handle_t *pamh, int flags, int argc,
                                 const char **argv) -> int
{
  return PAM_IGNORE;
}
PAM_EXTERN auto pam_sm_close_session(pam_handle_t *pamh, int flags, int argc,
                                     const char **argv) -> int
{
  return PAM_IGNORE;
}
PAM_EXTERN auto pam_sm_chauthtok(pam_handle_t *pamh, int flags, int argc,
                                 const char **argv) -> int
{
  return PAM_IGNORE;
}
PAM_EXTERN auto pam_sm_setcred(pam_handle_t *pamh, int flags, int argc,
                               const char **argv) -> int
{
  return PAM_IGNORE;
}

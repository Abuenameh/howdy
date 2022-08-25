#include <opencv2/core/utils/logger.hpp>

#include "../utils/argparse.hpp"

using namespace std::literals;

void add(argparse::Namespace &args, std::string &user);

void clear(argparse::Namespace &args, std::string &user);

void config();

void disable(argparse::Namespace &args);

void list(argparse::Namespace &args, std::string &user);

void remove(argparse::Namespace &args, std::string &user);

void set(argparse::Namespace &args);

void snapshot();

void test();

int main(int argc, char *argv[])
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    std::string user;

    // Try to get the original username (not "root") from shell
    char *user_p = getlogin();
    if (!user_p)
    {
        user_p = std::getenv("SUDO_USER");
    }

    // If that fails, try to get the direct user
    if (!user_p || ("root"s == user_p))
    {
        char *env_user = std::getenv("LOGNAME");
        if (!env_user)
            env_user = std::getenv("USER");
        if (!env_user)
            env_user = std::getenv("LNAME");
        if (!env_user)
            env_user = std::getenv("USERNAME");

        // If even that fails, error out
        if (!env_user || (""s == env_user))
        {
            std::cerr << "Could not determine user, please use the --user flag" << std::endl;
            exit(1);
        }
        else
        {
            user = env_user;
        }
    }
    else
    {
        user = user_p;
    }

    // Basic command setup
    argparse::ArgumentParser parser(argc, argv);
    parser.description("Command line interface for Howdy face authentication.")
        .formatter_class(argparse::RawDescriptionHelpFormatter)
        .add_help(false)
        .prog("howdy")
        .usage("howdy [-U USER] [--plain] [-h] [-y] command [arguments...]")
        .epilog("For support please visit\nhttps://github.com/boltgolt/howdy");

    // Add an argument for the command
    parser.add_argument("command")
        .help("The command option to execute, can be one of the following: add, clear, config, disable, list, remove, snapshot, set, test or version.")
        .metavar("command")
        .choices({"add", "clear", "config", "disable", "list", "remove", "set", "snapshot", "test", "version"});

    // Add an argument for the extra arguments of diable and remove
    parser.add_argument("arguments")
        .help("Optional arguments for the add, disable, remove and set commands.")
        .nargs("*");

    // Add the user flag
    parser.add_argument("-U", "--user")
        .default_value(user)
        .help("Set the user account to use.");

    // Add the -y flag
    parser.add_argument("-y")
        .help("Skip all questions.")
        .action("store_true");

    // Add the --plain flag
    parser.add_argument("--plain")
        .help("Print machine-friendly output.")
        .action("store_true");

    // Overwrite the default help message so we can use a uppercase S
    parser.add_argument("-h", "--help")
        .action("help")
        .default_value(argparse::SUPPRESS)
        .help("Show this help message and exit.");

    // If we only have 1 argument we print the help text
    if (argc < 2)
    {
        std::cout << "current active user: " << user << std::endl;
        parser.print_help();
        exit(0);
    }

    // Parse all arguments above
    argparse::Namespace args = parser.parse_args();

    user = args.get<std::string>("-U");

    // Check if we have rootish rights
    // This is this far down the file so running the command for help is always possible
    if (geteuid() != 0)
    {
        std::string args;
        for (int i = 1; i < argc; i++)
            args += " "s + argv[i];
        std::cerr << "Please run this command as root:" << std::endl;
        std::cerr << "\tsudo howdy" << args << std::endl;
        exit(1);
    }

    // Beyond this point the user can't change anymore, if we still have root as user we need to abort
    if (user == "root")
    {
        std::cerr << "Can't run howdy commands as root, please run this command with the --user flag" << std::endl;
        exit(1);
    }

    // Execute the right command
    std::string command = args.get<std::string>("command");
    if (command == "add")
        add(args, user);
    else if (command == "clear")
        clear(args, user);
    else if (command == "config")
        config();
    else if (command == "disable")
        disable(args);
    else if (command == "list")
        list(args, user);
    else if (command == "remove")
        remove(args, user);
    else if (command == "set")
        set(args);
    else if (command == "snapshot")
        snapshot();
    else if (command == "test")
        test();
    else
        std::cout << "Howdy 3.0.0 BETA" << std::endl;
}
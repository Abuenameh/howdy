#include <stdio.h>
#include <unistd.h>
#include <limits.h>

#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <map>
#include <iomanip>
#include <ctime>
#include <regex>

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <dlib/opencv.h>
#include <dlib/dnn.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include <INIReader.h>

#include "../video_capture.hh"
#include "../models.hh"
#include "../snapshot.hh"
#include "../rubber_stamps.hh"
#include "../utils.hh"
#include "../utils/string.hpp"

#include "../utils/json.hpp"
#include "../utils/argparse.hpp"

#define FMT_HEADER_ONLY
#include "../fmt/core.h"

using json = nlohmann::json;
using namespace dlib;
using namespace std::literals;

namespace fs = std::filesystem;

typedef std::chrono::time_point<std::chrono::system_clock> time_point;

void config()
{
    // Let the user know what we're doing
    std::cout << "Opening config.ini in the default editor" << std::endl;

    // Default to the nano editor
    std::string editor = "/bin/nano";

    // Use the user preferred editor if available
    if (std::getenv("EDITOR"))
        editor = std::getenv("EDITOR");
    else if (fs::is_regular_file(fs::status("/etc/alternatives/editor")))
        editor = "/etc/alternatives/editor";

    // Open the editor as a subprocess and fork it
    std::string config_file(PATH + "/config.ini");
    char *const args[] = {editor.data(), config_file.data(), NULL};
    execvp(editor.c_str(), args);
}
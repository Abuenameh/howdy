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
// #include <dlib/image_processing.h>

#include <INIReader.h>

#include "../video_capture.hh"
#include "../models.hh"
#include "../compare.hh"
#include "../snapshot.hh"
#include "../rubber_stamps.hh"
#include "../utils.hh"
#include "../string_utils/string_utils.hh"

#include "../utils/json.hpp"
#include "../utils/argparse.hpp"

#define FMT_HEADER_ONLY
#include "../fmt/core.h"

using json = nlohmann::json;
using namespace dlib;
using namespace std::literals;

namespace fs = std::filesystem;

typedef std::chrono::time_point<std::chrono::system_clock> time_point;

void clear(argparse::Namespace &args, std::string &user)
{
    // Check if the models folder is there
    if (!fs::exists(fs::status(PATH + "/models")))
    {
        std::cerr << "No models created yet, can't clear them if they don't exist" << std::endl;
        exit(1);
    }

    // Check if the user has a models file to delete
    if (!fs::is_regular_file(fs::status(fmt::format("{}/models/{}.dat", PATH, user))))
    {
        std::cerr << fmt::format("{} has no models or they have been cleared already", user) << std::endl;
        exit(1);
    }

    std::string ans;

    // Only ask the user if there's no -y flag
    if (args.get<bool>("y"))
    {
        // Double check with the user
        std::cout << "This will clear all models for " << user << std::endl;
        std::cout << "Do you want to continue [y/N]: ";
        std::getline(std::cin, ans);

        // Abort if they don't answer y or Y
        if (tolower(ans) != "y")
        {
            std::cout << std::endl
                      << "Inerpeting as a \"NO\", aborting" << std::endl;
            exit(1);
        }
    }

    // Delete otherwise
    fs::remove(fmt::format("{}/models/{}.dat", PATH, user));
    std::cout << std::endl
              << "Models cleared" << std::endl;
}
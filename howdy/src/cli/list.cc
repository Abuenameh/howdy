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
#include "../fmt/chrono.h"

using json = nlohmann::json;
using namespace dlib;
using namespace std::literals;

namespace fs = std::filesystem;

typedef std::chrono::time_point<std::chrono::system_clock> time_point;

void list(argparse::Namespace &args, std::string &user)
{
    // Check if the models file has been created yet
    if (!fs::exists(fs::status(PATH + "/models")))
    {
        std::cerr << "Face models have not been initialized yet, please run:" << std::endl;
        std::cerr << std::endl
                  << fmt::format("\tsudo howdy -U {} add", user) << std::endl
                  << std::endl;
        exit(1);
    }

    // Path to the models file
    std::string enc_file = PATH + "/models/" + user + ".dat";

    // Try to load the models file and abort if the user does not have it yet
    json encodings;
    if (fs::exists(fs::status(enc_file)))
    {
        std::ifstream f(enc_file);
        encodings = json::parse(f);
    }
    else
    {
        std::cerr << fmt::format("No face model known for the user {}, please run:", user) << std::endl;
        std::cerr << std::endl
                  << fmt::format("\tsudo howdy -U {} add", user) << std::endl
                  << std::endl;
        exit(1);
    }

    // Print a header if we're not in plain mode
    bool plain = args.get<bool>("plain");
    if (!plain)
    {
        std::cout << fmt::format("Known face models for {}:", user) << std::endl;
        std::cout << "\n\033[1;29m"
                  << "ID  Date                 Label\033[0m" << std::endl;
    }

    // Loop through all encodings and print info about them
    for (auto &enc : encodings)
    {
        // Start with the id
        std::cout << std::to_string(enc["id"].get<int>());

        // Add comma for machine reading
        if (plain)
            std::cout << ",";
        // Print padding spaces after the id for a nice layout
        else
            std::cout << std::string(4 - std::to_string(enc["id"].get<int>()).size(), ' ');

        // Format the time as ISO in the local timezone
        time_point time = time_point() + std::chrono::seconds(enc["time"].get<int>());
        std::cout << fmt::format("{:%Y-%m-%d %H:%M:%S}", time);
        // print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(enc["time"])), end="")

        // Seperate with commas again for machines, spaces otherwise
        std::cout << (plain ? "," : "  ");

        // End with the label
        std::cout << enc["label"].get<std::string>() << std::endl;
    }

    // Add a closing enter
    std::cout << std::endl;
}
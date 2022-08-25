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

void remove(argparse::Namespace &args, std::string &user)
{
    // Check if enough arguments have been passed
    std::vector<std::string> arguments = args.get<std::vector<std::string>>("arguments");
    if (arguments.empty())
    {
        std::cerr << "Please add the ID of the model you want to remove as an argument" << std::endl;
        std::cerr << "For example:" << std::endl;
        std::cerr << std::endl
                  << "\thowdy remove 0" << std::endl
                  << std::endl;
        std::cerr << "You can find the IDs by running:" << std::endl;
        std::cerr << std::endl
                  << "\thowdy list" << std::endl
                  << std::endl;
        exit(1);
    }

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

    // Tracks if a encoding with that id has been found
    bool found = false;

    // Get the ID from the cli arguments
    std::string id = arguments[0];

    // Loop though all encodings and check if they match the argument
    for (auto &enc : encodings)
    {
        if (std::to_string(enc["id"].get<int>()) == id)
        {
            // Only ask the user if there's no -y flag
            if (!args.get<bool>("y"))
            {
                // Double check with the user
                std::cout << fmt::format(
                                 "This will remove the model called \"{}\" for {}", enc["label"].get<std::string>(), user)
                          << std::endl;
                std::cout << "Do you want to continue [y/N]: ";
                std::string ans;
                std::getline(std::cin, ans);

                // Abort if the answer isn't yes
                if (tolower(ans) != "y")
                {
                    std::cout << std::endl
                              << "Interpreting as a \"NO\", aborting" << std::endl;
                    exit(1);
                }

                // Add a padding empty  line
                std::cout << std::endl;
            }

            // Mark as found and print an enter
            found = true;
            break;
        }
    }

    // Abort if no matching id was found
    if (!found)
    {
        std::cout << fmt::format("No model with ID {} exists for {}", id, user) << std::endl;
        exit(1);
    }

    // Remove the entire file if this encoding is the only one
    if (encodings.size() == 1)
    {
        fs::remove(PATH + "/models/" + user + ".dat");
        std::cout << "Removed last model, howdy disabled for user" << std::endl;
    }
    else
    {
        // A place holder to contain the encodings that will remain
        json new_encodings;

        // Loop though all encodings and only add those that don't need to be removed
        for (auto enc : encodings)
        {
            if (std::to_string(enc["id"].get<int>()) != id)
                new_encodings.push_back(enc);
        }

        // Save this new set to disk
        std::ofstream datafile(enc_file);
        datafile << new_encodings;

        std::cout << fmt::format("Removed model {}", id) << std::endl;
    }
}
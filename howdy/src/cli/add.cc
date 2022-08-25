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

#include "../utils/json.hpp"
#include "../utils/argparse.hpp"

#define FMT_HEADER_ONLY
#include "../fmt/core.h"

using json = nlohmann::json;
using namespace dlib;
using namespace std::literals;

namespace fs = std::filesystem;

typedef std::chrono::time_point<std::chrono::system_clock> time_point;

void add(argparse::Namespace &args, std::string &user)
{
    // Test if at lest 1 of the data files is there and abort if it's not
    if (!fs::exists(fs::status(PATH + "/dlib-data/shape_predictor_5_face_landmarks.dat")))
    {
        std::cerr << "Data files have not been downloaded, please run the following commands:" << std::endl;
        std::cerr << std::endl
                  << fmt::format("\tcd {}/dlib-data", PATH) << std::endl;
        std::cerr << "\tsudo ./install.sh" << std::endl
                  << std::endl;
        exit(1);
    }

    // Read config from disk
    INIReader config(PATH + "/config.ini");

    bool use_cnn = config.GetBoolean("core", "use_cnn", false);
    face_detection_model *face_detector_p;
    if (use_cnn)
    {
        face_detector_p = new cnn_face_detection_model_v1(PATH + "/dlib-data/mmod_human_face_detector.dat");
    }
    else
    {
        face_detector_p = new frontal_face_detector_model();
    }
    face_detection_model &face_detector = *face_detector_p;

    shape_predictor_model pose_predictor = shape_predictor_model(PATH + "/dlib-data/shape_predictor_5_face_landmarks.dat");
    face_recognition_model_v1 face_encoder = face_recognition_model_v1(PATH + "/dlib-data/dlib_face_recognition_resnet_model_v1.dat");

    // The permanent file to store the encoded model in
    std::string enc_file = PATH + "/models/" + user + ".dat";
    // Known encodings
    json encodings;

    // Make the ./models folder if it doesn't already exist
    if (!fs::exists(fs::status(PATH + "/models")))
    {
        std::cout << "No face model folder found, creating one" << std::endl;
        fs::create_directories(PATH + "/models");
    }

    // To try read a premade encodings file if it exists
    try {
    std::ifstream f(enc_file);
    encodings = json::parse(f);
    }
    catch (std::exception &e) {
    }

    // Print a warning if too many encodings are being added
    if (encodings.size() > 3)
    {
        std::cout << "NOTICE: Each additional model slows down the face recognition engine slightly" << std::endl;
        std::cout << "Press Ctrl+C to cancel" << std::endl
                  << std::endl;
    }

    // Make clear what we are doing if not human
    if (args.get<bool>("plain"))
    {
        std::cout << "Adding face model for the user " << user << std::endl;
    }

    // Set the default label
    std::string label = "Initial model";

    // Get the label from the cli arguments if provided
    if (args.exists("arguments"))
        label = args.get<std::vector<std::string>>("arguments")[0];

    // If models already exist, set that default label
    else if (!encodings.empty())
        label = "Model #" + std::to_string(encodings.size() + 1);

    std::string label_in;

    // Keep de default name if we can't ask questions
    if (args.get<bool>("y"))
        std::cout << fmt::format("Using default label {} because of -y flag", label);
    else
    {
        // Ask the user for a custom label
        std::cout << fmt::format("Enter a label for this new model [{}]: ", label);
        std::getline(std::cin, label_in);

        // Set the custom label (if any) and limit it to 24 characters
        if (label_in != "")
            label = label_in.substr(0, 24);
    }

    // Remove illegal characters
    if (label.find(",") != std::string::npos)
    {
        std::cout << "NOTICE: Removing illegal character \",\" from model name" << std::endl;
        label = std::regex_replace(label, std::regex(","), "");
    }

    // Prepare the metadata for insertion
    json insert_model = {
        {"time", std::chrono::time_point_cast<std::chrono::seconds>(now()).time_since_epoch().count()},
        {"label", label},
        {"id", encodings.size()},
        {"data", {}}
    };

    // Set up video_capture
    VideoCapture video_capture(config);

    std::cout << std::endl
              << "Please look straight into the camera" << std::endl;

    // Give the user time to read
    std::this_thread::sleep_for(2s);

    // Will contain found face encodings
    // enc = [];
    // Count the number of read frames
    int frames = 0;
    // Count the number of illuminated read frames
    int valid_frames = 0;
    // Count the number of illuminated frames that
    // were rejected for being too dark
    int dark_tries = 0;
    // Track the running darkness total
    double dark_running_total = 0;
    std::vector<rectangle> face_locations;

    double dark_threshold = config.GetReal("video", "dark_threshold", 50.0);

    auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));

    cv::Mat tempframe;
    cv::Mat frame, gsframe;

    // Loop through frames till we hit a timeout
    while (frames < 60)
    {
        frames += 1;
        // Grab a single frame of video
        video_capture.read_frame(frame, tempframe);
        clahe->apply(tempframe, gsframe);

        // Create a histogram of the image with 8 values
        cv::Mat hist;
        cv::calcHist(std::vector<cv::Mat>{gsframe}, std::vector<int>{0}, cv::Mat(), hist, std::vector<int>{8}, std::vector<float>{0, 256});
        // All values combined for percentage calculation
        double hist_total = cv::sum(hist)[0];

        // Calculate frame darkness
        double darkness = (hist.at<float>(0) / hist_total * 100);

        // If the image is fully black due to a bad camera read,
        // skip to the next frame
        if ((hist_total == 0) or (darkness == 100))
            continue;

        // Include this frame in calculating our average session brightness
        dark_running_total += darkness;
        valid_frames += 1;

        // If the image exceeds darkness threshold due to subject distance,
        // skip to the next frame
        if (darkness > dark_threshold)
        {
            dark_tries += 1;
            continue;
        }

        // Get all faces from that frame as encodings
        face_locations = face_detector(gsframe, 1);

        // If we've found at least one, we can continue
        if (!face_locations.empty())
            break;
    }

    video_capture.release();

    // If we've found no faces, try to determine why
    if (face_locations.empty())
    {
        if (valid_frames == 0)
            std::cerr << "Camera saw only black frames - is IR emitter working?" << std::endl;
        else if (valid_frames == dark_tries)
        {
            std::cerr << "All frames were too dark, please check dark_threshold in config" << std::endl;
            std::cerr << fmt::format("Average darkness: {}, Threshold: {}", std::to_string(dark_running_total / valid_frames), std::to_string(dark_threshold)) << std::endl;
        }
        else
            std::cerr << "No face detected, aborting" << std::endl;
        exit(1);
    }

    // If more than 1 faces are detected we can't know wich one belongs to the user
    else if (face_locations.size() > 1)
    {
        std::cerr << "Multiple faces detected, aborting" << std::endl;
        exit(1);
    }

    rectangle face_location = face_locations[0];

    // Get the encodings in the frame
    auto face_landmark = pose_predictor(frame, face_location);
    auto face_encoding = face_encoder.compute_face_descriptor(frame, face_landmark, 1);

    std::vector<double> encoding(face_encoding.begin(), face_encoding.end());
    insert_model["data"].push_back(encoding);

    // Insert full object into the list
    encodings.push_back(insert_model);

    // Save the new encodings to disk
    std::ofstream datafile(enc_file);
    datafile << encodings;

    // Give let the user know how it went
    std::cout << std::endl
              << "Scan complete" << std::endl
              << "Added a new model to " << user << std::endl;
}
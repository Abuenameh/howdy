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
#include <opencv2/highgui.hpp>
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

// Enable a delay in the loop
bool slow_mode = false;

/*Handle mouse events*/
void mouse(int event, int x, int y, int flags, void *param)
{
    // Toggle slowmode on click
    if (event == cv::EVENT_LBUTTONDOWN)
        slow_mode = !slow_mode;
}

void test()
{
    // Read config from disk
    INIReader config(PATH + "/config.ini");

    if (config.GetString("video", "recording_plugin", "") != "opencv")
    {
        std::cerr << "Howdy has been configured to use a recorder which doesn't support the test command yet, aborting" << std::endl;
        exit(12);
    }

    VideoCapture video_capture(config);

    // Read exposure and dark_thresholds from config to use in the main loop
    int exposure = config.GetInteger("video", "exposure", -1);
    double dark_threshold = config.GetReal("video", "dark_threshold", 50.0);

    // Let the user know what's up
    std::cout << R"(
Opening a window with a test feed

Press ctrl+C in this terminal to quit
Click on the image to enable or disable slow mode
)" << std::endl;

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

    auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));

    // Open the window and attach a a mouse listener
    cv::namedWindow("Howdy Test");
    cv::setMouseCallback("Howdy Test", mouse);

    // Count all frames ever
    int total_frames = 0;
    // Count all frames per second
    int sec_frames = 0;
    // Last secands FPS
    int fps = 0;
    // The current second we're counting
    int sec = std::chrono::time_point_cast<std::chrono::seconds>(now()).time_since_epoch().count();
    // sec = int(time.time())
    // recognition time
    int rec_tm = 0;

    // Wrap everything in an keyboard interupt handler
    try
    {
        while (true)
        {
            int frame_tm = std::chrono::time_point_cast<std::chrono::seconds>(now()).time_since_epoch().count();

            // Increment the frames
            total_frames += 1;
            sec_frames += 1;

            // If we've entered a new second
            if (sec != frame_tm)
            {
                // Set the last seconds FPS
                fps = sec_frames;

                // Set the new second and reset the counter
                sec = frame_tm;
                sec_frames = 0;
            }

            // Grab a single frame of video
            cv::Mat tempframe;
            cv::Mat ret, frame;
            video_capture.read_frame(ret, tempframe);

            clahe->apply(tempframe, frame);
            // Make a frame to put overlays in
            cv::Mat tempoverlay = frame;
            cv::Mat overlay;
            cv::cvtColor(tempoverlay, overlay, cv::COLOR_GRAY2BGR);

            // Fetch the frame height and width
            int height = frame.rows;
            int width = frame.cols;

            // Create a histogram of the image with 8 values
            cv::Mat hist;
            cv::calcHist(std::vector<cv::Mat>{frame}, std::vector<int>{0}, cv::Mat(), hist, std::vector<int>{8}, std::vector<float>{0, 256});
            // All values combined for percentage calculation
            int hist_total = int(cv::sum(hist)[0]);
            // Fill with the overal containing percentage
            std::vector<double> hist_perc;

            // Loop though all values to calculate a percentage and add it to the overlay
            for (int index = 0; index < hist.rows; index++)
            {
                double value_perc = hist.at<float>(index, 0) / hist_total * 100;
                hist_perc.push_back(value_perc);

                // Top left pont, 10px margins
                cv::Point p1 = cv::Point(20 + (10 * index), 10);
                // Bottom right point makes the bar 10px thick, with an height of half the percentage
                cv::Point p2 = cv::Point(10 + (10 * index), int(value_perc / 2 + 10));
                // Draw the bar in green
                cv::rectangle(overlay, p1, p2, cv::Scalar(0, 200, 0), cv::FILLED);
            }

            /*Print the status text by line number*/
            std::function<void(int, std::string)> print_text{[&](int line_number, std::string text)
                                                             {
                                                                 cv::putText(overlay, text, cv::Point(10, height - 10 - (10 * line_number)), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 255, 0), 0, cv::LINE_AA);
                                                             }};

            // Print the statis in the bottom left
            print_text(0, fmt::format("RESOLUTION: {}x{}", height, width));
            print_text(1, fmt::format("FPS: {}", fps));
            print_text(2, fmt::format("FRAMES: {}", total_frames));
            print_text(3, fmt::format("RECOGNITION: {}ms", std::round(rec_tm * 1000)));

            // Show that slow mode is on, if it's on
            if (slow_mode)
                cv::putText(overlay, "SLOW MODE", cv::Point(width - 66, height - 10), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 255), 0, cv::LINE_AA);

            // Ignore dark frames
            if (hist_perc[0] > dark_threshold)
            {
                // Show that this is an ignored frame in the top right
                cv::putText(overlay, "DARK FRAME", cv::Point(width - 68, 16), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 255), 0, cv::LINE_AA);
            }
            else
            {
                // Show that this is an active frame
                cv::putText(overlay, "SCAN FRAME", cv::Point(width - 68, 16), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 255, 0), 0, cv::LINE_AA);

                rec_tm = std::chrono::time_point_cast<std::chrono::seconds>(now()).time_since_epoch().count();
                // Get the locations of all faces and their locations
                // Upsample it once
                std::vector<rectangle> face_locations = face_detector(frame, 1);
                rec_tm = std::chrono::time_point_cast<std::chrono::seconds>(now()).time_since_epoch().count() - rec_tm;

                // Loop though all faces and paint a circle around them
                for (auto &loc : face_locations)
                {
                    // Get the center X and Y from the rectangular points
                    int x = int((loc.right() - loc.left()) / 2) + loc.left();
                    int y = int((loc.bottom() - loc.top()) / 2) + loc.top();

                    // Get the raduis from the with of the square
                    int r = (loc.right() - loc.left()) / 2;
                    // Add 20% padding
                    r = int(r + (r * 0.2));

                    // Draw the Circle in green
                    cv::circle(overlay, cv::Point(x, y), r, cv::Scalar(0, 0, 230), 2);
                }
            }

            // Add the overlay to the frame with some transparency
            double alpha = 0.65;
            cv::Mat colorframe;
            cv::cvtColor(frame, colorframe, cv::COLOR_GRAY2BGR);
            cv::addWeighted(overlay, alpha, colorframe, 1 - alpha, 0, frame);

            // Show the image in a window
            cv::imshow("Howdy Test", frame);

            // Quit on any keypress
            if (cv::waitKey(1) != -1)
                throw std::runtime_error("");

            int frame_time = std::chrono::time_point_cast<std::chrono::seconds>(now()).time_since_epoch().count() - frame_tm;

            // Delay the frame if slowmode is on
            if (slow_mode)
                std::this_thread::sleep_for(std::chrono::seconds(int(std::max(0.5 - frame_time, 0.0))));

            if (exposure != -1)
            {
                // For a strange reason on some cameras (e.g. Lenoxo X1E)
                // setting manual exposure works only after a couple frames
                // are captured and even after a delay it does not
                // always work. Setting exposure at every frame is
                // reliable though.
                video_capture.set(cv::CAP_PROP_AUTO_EXPOSURE, 1.0); // 1 = Manual
                video_capture.set(cv::CAP_PROP_EXPOSURE, double(exposure));
            }
        }
    }
    catch (...)
    {
        // # On ctrl+C
        // except KeyboardInterrupt:
        // Let the user know we're stopping
        std::cout << std::endl
                  << "Closing window" << std::endl;

        // Release handle to the webcam
        cv::destroyAllWindows();
    }
}
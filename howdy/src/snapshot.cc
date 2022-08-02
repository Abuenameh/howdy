#include <iostream>
#include <iomanip>
#include <ctime>
#include <filesystem>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "utils.hh"
#include "snapshot.hh"

namespace fs = std::filesystem;

std::string generate(std::vector<cv::Mat> &frames, std::vector<std::string> &text_lines)
{
    // Don't execute if no frames were given
    if (frames.size() == 0)
        return "";

    // Get frame dimensions
    int frame_height = frames[0].rows;
    int frame_width = frames[0].cols;
    // Spread the given frames out horizontally
    cv::Mat snap;
    cv::hconcat(frames, snap);

    // Create colors
    cv::Scalar pad_color(44, 44, 44);
    cv::Scalar text_color(255, 255, 255);

    // Add a gray square at the bottom of the image
    cv::Mat temp;
    cv::copyMakeBorder(snap, temp, 0, text_lines.size() * 20 + 40, 0, 0, cv::BORDER_CONSTANT, pad_color);
    snap = temp;

    // Add the Howdy logo if there's space to do so
    if (frames.size() > 1)
    {
        // Load the logo from file
        cv::Mat logo = cv::imread(PATH + "/logo.png");
        // Calculate the position of the logo
        int logo_y = frame_height + 20;
        int logo_x = frame_width * frames.size() - 210;

        // Overlay the logo on top of the image
        snap(cv::Range(logo_y, logo_y + 57), cv::Range(logo_x, logo_x + 180)) = logo;
    }

    // Go through each line
    int line_number = 0;
    for (auto &line : text_lines)
    {
        // Calculate how far the line should be from the top
        int padding_top = frame_height + 30 + (line_number * 20);
        // Print the line onto the image
        cv::putText(snap, line, cv::Point(30, padding_top), cv::FONT_HERSHEY_SIMPLEX, 0.4, text_color, 0, cv::LINE_AA);

        line_number += 1;
    }

    // Made sure a snapshot folder exist
    if (!fs::exists(fs::status(PATH + "/snapshots")))
    {
        fs::create_directories(PATH + "/snapshots");
    }

    // Generate a filename based on the current time
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    std::ostringstream osstream;
    osstream << std::put_time(&tm, "%Y%m%dT%H%M%S.jpg");
    std::string filename = osstream.str();
    // Write the image to that file
    cv::imwrite(PATH + "/snapshots/" + filename, snap);

    // Return the saved file location
    return PATH + "/snapshots/" + filename;
}

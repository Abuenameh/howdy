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

void snapshot()
{
// Read the config
    INIReader config(PATH + "/config.ini");

// Start video capture
VideoCapture video_capture(config);

// Read a frame to activate emitters
    cv::Mat frame, gsframe;
video_capture.read_frame(frame, gsframe);

// Read exposure and dark_thresholds from config to use in the main loop
int exposure = config.GetInteger("video", "exposure", -1);
double dark_threshold = config.GetReal("video", "dark_threshold", 50);

// Collection of recorded frames
std::vector<cv::Mat> frames;

while (true) {
	// Grab a single frame of video
	video_capture.read_frame(frame, gsframe);

	// Add the frame to the list
	frames.push_back(frame);

	// Stop the loop if we have 4 frames
	if (frames.size() >= 4)
		break;
}

// Generate a snapshot image from the frames
std::vector<std::string> text_lines{
	"GENERATED SNAPSHOT",
	"Date: " + fmt::format("{:%Y/%m/%d %H:%M:%S} UTC", now()),
	"Dark threshold config: " + fmt::format("{:.1f}", config.GetReal("video", "dark_threshold", 50.0)),
	"Certainty config: " + fmt::format("{:.1f}", config.GetReal("video", "certainty", 3.5))
};
std::string file = generate(frames, text_lines);

// Show the file location in console
std::cout << "Generated snapshot saved as" << std::endl;
std::cout << file << std::endl;

}
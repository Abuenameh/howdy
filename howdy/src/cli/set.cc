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

void set(argparse::Namespace &args)
{
// Get the absolute filepath
std::string config_path(PATH + "/config.ini");

// Check if enough arguments have been passed
    std::vector<std::string> arguments = args.get<std::vector<std::string>>("arguments");
if (arguments.size() < 2) {
	std::cerr << "Please add a setting you would like to change and the value to set it to" << std::endl;
	std::cerr << "For example:" << std::endl;
	std::cerr << std::endl << "\thowdy set certainty 3" << std::endl << std::endl;
	exit(1);
}

// Get the name and value from the cli
std::string set_name = arguments[0];
std::string set_value = arguments[1];

// Will be filled with the correctly config line to update
std::string found_line;

// Loop through all lines in the config file
std::ifstream config_file(config_path);
std::string line;
while (std::getline(config_file, line)) {
	// Save the line if it starts with the requested config option
	if (line.starts_with(set_name + " "))
		found_line = line;
}
config_file.close();

// If we don't have the line it is not in the config file
if (found_line == "") {
	std::cerr << fmt::format("Could not find a \"{}\" config option to set", set_name) << std::endl;
	exit(1);
}

// Go through the file again and update the correct line
std::string config_tmp_path(PATH + "/config.ini.tmp");
std::ofstream tmp_file(config_tmp_path);
std::regex set_regex(found_line);
std::string replacement = set_name + " = " + set_value;
config_file.open(config_path);
while (std::getline(config_file, line)) {
    line = std::regex_replace(line, set_regex, replacement);
    tmp_file << line << std::endl;
}
fs::rename(config_tmp_path, config_path);

std::cout << "Config option updated" << std::endl;

}
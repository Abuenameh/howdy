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
// #include <dlib/image_processing.h>

#include <INIReader.h>

#include "../video_capture.hh"
#include "../models.hh"
#include "../snapshot.hh"
#include "../rubber_stamps.hh"
#include "../utils.hh"
#include "../utils/string.hpp"

#include "../utils/json.hpp"
#include "../utils/argparse.hpp"
#include "../process/process.hpp"

#define FMT_HEADER_ONLY
#include "../fmt/core.h"

using json = nlohmann::json;
using namespace dlib;
using namespace std::literals;
using namespace TinyProcessLib;

namespace fs = std::filesystem;

typedef std::chrono::time_point<std::chrono::system_clock> time_point;

void disable(argparse::Namespace &args)
{
	// Get the absolute filepath
	std::string config_path(PATH + "/config.ini");

	// Read config from disk
	INIReader config(config_path);

	// Check if enough arguments have been passed
	std::vector<std::string> arguments = args.get<std::vector<std::string>>("arguments");
	if (arguments.empty())
	{
		std::cerr << "Please add a 0 (enable) or a 1 (disable) as an argument" << std::endl;
		exit(1);
	}

	// Get the cli argument
	std::string argument = arguments[0];

	// Translate the argument to the right string
	std::string out_value;
	if (argument == "1" || tolower(argument) == "true")
		out_value = "true";
	else if (argument == "0" || tolower(argument) == "false")
		out_value = "false";
	else
	{
		// If it's not a 0 or a 1, it's invalid
		std::cerr << "Please only use 0 (enable) or 1 (disable) as an argument" << std::endl;
		exit(1);
	}

	// Don't do anything when the state is already the requested one
	if (out_value == config.GetString("core", "disabled", ""))
	{
		std::cerr << "The disable option has already been set to " << out_value << std::endl;
		exit(1);
	}

	// Loop though the config file and only replace the line containing the disable config
	std::string config_tmp_path(PATH + "/config.ini.tmp");
	std::ifstream config_file(config_path);
	std::ofstream tmp_file(config_tmp_path);
	std::string line;
	std::regex disabled_regex("disabled = " + config.GetString("core", "disabled", ""));
	std::string replacement = "disabled = " + out_value;
	while (std::getline(config_file, line))
	{
		line = std::regex_replace(line, disabled_regex, replacement);
		tmp_file << line << std::endl;
	}
	fs::rename(config_tmp_path, config_path);

	// Print what we just did
	if (out_value == "true")
		std::cout << "Howdy has been disabled" << std::endl;
	else
		std::cout << "Howdy has been enabled" << std::endl;
}
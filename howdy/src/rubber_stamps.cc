#include <sys/syslog.h>
#include <syslog.h>

#include <memory>
#include <regex>

#include "string_utils/string_utils.hh"

#include "utils.hh"
#include "rubber_stamps.hh"

using namespace std::chrono_literals;

enum TextType
{
	UI_TEXT,
	UI_SUBTEXT
};

typedef std::variant<std::string, int, double, bool> option;

/* Howdy rubber stamp */
class RubberStamp
{

public:
	RubberStamp(bool verbose, INIReader &config_, std::shared_ptr<Process> gtk_proc_, OpenCV &opencv_) : verbose(verbose), config(config_), gtk_proc(gtk_proc_), opencv(opencv_) {}

	virtual ~RubberStamp() = default;

	/* Convert an ui string to input howdy-gtk understands */
	void set_ui_text(std::string text, TextType type)
	{
		std::string typedec = "M";

		if (type == UI_SUBTEXT)
			typedec = "S";

		send_ui_raw(typedec + "=" + text);
	}

	/* Write raw command to howdy-gtk stdin */
	void send_ui_raw(std::string command)
	{
		if (config.GetBoolean("debug", "verbose_stamps", false))
			syslog(LOG_INFO, "Sending command to howdy-gtk: %s", command.c_str());

		// Add a newline because the ui reads per line
		command += " \n";

		// If we're connected to the ui
		if (gtk_proc)
		{
			// Send the command as bytes
			gtk_proc->write(command);

			// Write a padding line to force the command through any buffers
			gtk_proc->write("P=_PADDING \n");
		}
	}

	virtual std::string name()
	{
		return "";
	}

	virtual void declare_config() {}

	virtual bool run()
	{
		return true;
	}

	bool verbose;
	INIReader &config;
	std::shared_ptr<Process> gtk_proc;
	OpenCV &opencv;
	std::map<std::string, option> options;
};

class nod : public RubberStamp
{
public:
	nod(bool verbose, INIReader &config, std::shared_ptr<Process> gtk_proc, OpenCV &opencv) : RubberStamp(verbose, config, gtk_proc, opencv) {}

	virtual ~nod() = default;

	virtual std::string name()
	{
		return "nod";
	}

	/* Set the default values for the optional arguments */
	virtual void declare_config()
	{
		options["min_distance"] = 6.0;
		options["min_directions"] = 2;
	}

	/* Track a users nose to see if they nod yes or no */
	virtual bool run()
	{
		set_ui_text("Nod to confirm", UI_TEXT);
		set_ui_text("Shake your head to abort", UI_SUBTEXT);

		// Stores relative distance between the 2 eyes in the last frame
		// Used to calculate the distance of the nose traveled in relation to face size in the frame
		double last_reldist = -1;
		// Last point the nose was at
		std::map<std::string, long> last_nosepoint{{"x", -1}, {"y", -1}};
		// Contains booleans recording successful nods and their directions
		std::map<std::string, std::vector<bool>> recorded_nods{{"x", std::vector<bool>()}, {"y", std::vector<bool>()}};

		time_point starttime = now();

		// Keep running the loop while we have not hit timeout yet
		while (now() < starttime + std::chrono::seconds(int(round(std::get<double>(options["timeout"])))))
		{
			// Read a frame from the camera
			cv::Mat ret, tempframe;
			cv::Mat frame;
			opencv.video_capture.read_frame(ret, tempframe);

			// Apply CLAHE to get a better picture
			opencv.clahe->apply(tempframe, frame);

			// Detect all faces in the frame
			std::vector<rectangle> face_locations = opencv.face_detector(frame, 1);

			// Only continue if exacty 1 face is visible in the frame
			if (face_locations.size() != 1)
				continue;

			// Get the position of the eyes and tip of the nose
			auto face_landmarks = opencv.pose_predictor(frame, face_locations[0]);

			// Calculate the relative distance between the 2 eyes
			double reldist = face_landmarks.part(0).x() - face_landmarks.part(2).x();
			// Avarage this out with the distance found in the last frame to smooth it out
			double avg_reldist = (last_reldist + reldist) / 2;

			// Calulate horizontal movement (shaking head) and vertical movement (nodding)
			for (std::string axis : {"x", "y"})
			{
				// Get the location of the nose on the active axis
				long nosepoint = (axis == "x") ? face_landmarks.part(4).x() : face_landmarks.part(4).y();
				// nosepoint = getattr(face_landmarks.part(4), axis)

				// If this is the first frame set the previous values to the current ones
				if (last_nosepoint[axis] == -1)
				{
					last_nosepoint[axis] = nosepoint;
					last_reldist = reldist;
				}

				double mindist = std::get<double>(options["min_distance"]);
				// Get the relative movement by taking the distance traveled and deviding it by eye distance
				double movement = (nosepoint - last_nosepoint[axis]) * 100 / std::max(avg_reldist, 1.0);

				// If the movement is over the minimal distance threshold
				if (movement < -mindist || movement > mindist)
				{
					// If this is the first recorded nod, add it to the array
					if (recorded_nods[axis].size() == 0)
						recorded_nods[axis].push_back(movement < 0);

					// Otherwise, only add this nod if the previous nod with in the other direction
					else if (recorded_nods[axis].back() != (movement < 0))
						recorded_nods[axis].push_back(movement < 0);
				}

				// Check if we have nodded enough on this axis
				if (recorded_nods[axis].size() >= std::get<int>(options["min_directions"]))
				{
					// If nodded yes, show confirmation in ui
					if (axis == "y")
						set_ui_text("Confirmed authentication", UI_TEXT);
					// If shaken no, show abort message
					else
						set_ui_text("Aborted authentication", UI_TEXT);

					// 	Remove subtext
					set_ui_text("", UI_SUBTEXT);

					// 	Return true for nodding yes and false for shaking no
					std::this_thread::sleep_for(800ms);
					return (axis == "y");
				}

				// Save the relative distance and the nosepoint for next loop
				last_reldist = reldist;
				last_nosepoint[axis] = nosepoint;
			}
		}

		// We've fallen out of the loop, so timeout has been hit
		return !std::get<bool>(options["failsafe"]);
	}
};

class hotkey : public RubberStamp {
	public:
	hotkey(bool verbose, INIReader &config, std::shared_ptr<Process> gtk_proc, OpenCV &opencv) : RubberStamp(verbose, config, gtk_proc, opencv) {
		pressed_key = "none";
	}

	virtual ~hotkey() = default;

	virtual std::string name()
	{
		return "hotkey";
	}

		/*Set the default values for the optional arguments*/
	virtual void declare_config() {
		options["abort_key"] = "esc";
		options["confirm_key"] = "enter";
	}

		/*Wait for the user to press a hotkey*/
	virtual bool run() {
		double time_left = std::get<double>(options["timeout"]);
		std::string time_string = std::get<bool>(options["failsafe"]) ? "Aborting authorisation in " : "Authorising in ";

		// Set the ui to default strings
		set_ui_text(time_string + std::to_string(int(time_left)), UI_TEXT);
		set_ui_text("Press " + std::get<std::string>(options["abort_key"]) + " to abort, " + std::get<std::string>(options["confirm_key"]) + " to authorise", UI_SUBTEXT);

		// Register hotkeys with the kernel
		add_hotkey(std::get<std::string>(options["abort_key"]), [&]() {
			on_key("abort");
			return false;
		});
		add_hotkey(std::get<std::string>(options["confirm_key"]), [&]() {
			on_key("confirm");
			return false;
		});

		// While we have not hit our timeout yet
		while (time_left > 0) {
			// Remove 0.1 seconds from the timer, as that's how long we sleep
			time_left -= 0.1;
			// Update the ui with the new time
			set_ui_text(time_string + std::to_string(int(time_left) + 1), UI_TEXT);

			// If the abort key was pressed while the loop was sleeping
			if (pressed_key == "abort") {
				// Set the ui to confirm the abort
				set_ui_text("Authentication aborted", UI_TEXT);
				set_ui_text("", UI_SUBTEXT);

				// Exit
				std::this_thread::sleep_for(1s);
				return false;
			}

			// If confirm has pressed, return that auth can continue
			else if (pressed_key == "confirm")
				return true;

			// If no key has been pressed, wait for a bit and check again
			std::this_thread::sleep_for(100ms);
		}

		// When our timeout hits, either abort or continue based on failsafe of faildeadly
		return !std::get<bool>(options["failsafe"]);
	}

		/*Called when the user presses a key*/
	void on_key(std::string type) {
		pressed_key = type;
	}

	std::string pressed_key;

};

std::vector<std::shared_ptr<RubberStamp>> get_installed_stamps(bool verbose, INIReader &config, std::shared_ptr<Process> gtk_proc, OpenCV &opencv)
{
	std::vector<std::shared_ptr<RubberStamp>> installed_stamps;
	installed_stamps.push_back(std::shared_ptr<RubberStamp>(new nod(verbose, config, gtk_proc, opencv)));
	installed_stamps.push_back(std::shared_ptr<RubberStamp>(new hotkey(verbose, config, gtk_proc, opencv)));
	return installed_stamps;
}

void execute(INIReader &config, std::shared_ptr<Process> gtk_proc, OpenCV &opencv)
{
	bool verbose = config.GetBoolean("debug", "verbose_stamps", false);
	std::vector<std::shared_ptr<RubberStamp>> installed_stamps = get_installed_stamps(verbose, config, gtk_proc, opencv);

	std::vector<std::string> stamp_names;
	std::transform(begin(installed_stamps), end(installed_stamps), std::back_inserter(stamp_names), [](std::shared_ptr<RubberStamp> stamp)
				   { return stamp->name(); });
	std::map<std::string, std::shared_ptr<RubberStamp>> stamp_map;
	std::transform(begin(installed_stamps), end(installed_stamps), std::inserter(stamp_map, stamp_map.end()), [](std::shared_ptr<RubberStamp> stamp)
				   { return std::pair{stamp->name(), stamp}; });

	if (verbose)
	{
		std::ostringstream osstream;
		if (!stamp_names.empty())
		{
			std::copy(stamp_names.begin(), stamp_names.end() - 1, std::ostream_iterator<std::string>(osstream, ","));
			osstream << stamp_names.back();
		}
		syslog(LOG_INFO, "Installed rubberstamps: %s", osstream.str().c_str());
	}

	// Get the rules defined in the config
	std::string raw_rules = config.GetString("rubberstamps", "stamp_rules", "");
	std::vector<std::string> rules = split(raw_rules, "\n");

	// Go through the rules one by one
	for (auto rule : rules)
	{
		trim(rule);

		if (rule.length() <= 1)
			continue;

		// Parse the rule with regex
		std::regex rule_regex("^(\\w+)\\s+([\\w\\.]+)\\s+([a-z]+)(.*)?$", std::regex::ECMAScript | std::regex::icase);

		// Error out if the regex did not match (invalid line)
		std::smatch regex_result;
		if (!std::regex_search(rule, regex_result, rule_regex))
		{
			syslog(LOG_ERR, "Error parsing rubberstamp rule: %s", rule.c_str());
			continue;
		}

		std::string type = regex_result[1];

		// Error out if the stamp name in the rule is not a file
		if (std::find(begin(stamp_names), end(stamp_names), type) == end(stamp_names))
		{
			syslog(LOG_ERR, "Stamp not installed: %s", type.c_str());
			continue;
		}

		// Try to get the class with the same name
		std::shared_ptr<RubberStamp> instance = stamp_map[type];

		// // Set some opensv shorthands
		// instance.video_capture = opencv["video_capture"]
		// instance.face_detector = opencv["face_detector"]
		// instance.pose_predictor = opencv["pose_predictor"]
		// instance.clahe = opencv["clahe"]

		// // Parse and set the 2 required options for all rubberstamps
		instance->options["timeout"] = std::stod(std::regex_replace(regex_result[2].str(), std::regex("[a-zA-Z]"), ""));
		instance->options["failsafe"] = regex_result[3] != "faildeadly";

		// Try to get the class do declare its other config variables
		instance->declare_config();

		// Split the optional arguments at the end of the rule by spaces
		std::vector<std::string> raw_options = split(regex_result[4].str(), " \t,");

		// For each of those optional arguments
		for (std::string &raw_option : raw_options)
		{
			// Get the key to the left, and the value to the right of the equal sign
			std::vector<std::string> split_option = split(raw_option, "=");
			std::string key = split_option[0];
			std::string value_str = split_option[1];
			option value = value_str;

			// Error out if a key has been set that was not declared by the module before
			if (!instance->options.contains(key))
			{
				syslog(LOG_ERR, "Unknow config option for rubberstamp %s: %s", type.c_str(), key.c_str());
				continue;
			}

			// Convert the argument string to an int or float if the declared option has that type
			if (std::holds_alternative<int>(instance->options[key]))
				value = std::stoi(value_str);
			else if (std::holds_alternative<double>(instance->options[key]))
				value = std::stod(value_str);

			instance->options[key] = value;
		}

		if (verbose)
		{
			syslog(LOG_INFO, "Stamp \"%s\" options parsed:", type.c_str());
			for (auto &opt_pair : instance->options)
			{
				option opt = opt_pair.second;
				std::string value;
				if (std::holds_alternative<std::string>(opt))
					value = std::get<std::string>(opt);
				else if (std::holds_alternative<int>(opt))
					value = std::to_string(std::get<int>(opt));
				else if (std::holds_alternative<double>(opt))
					value = std::to_string(std::get<double>(opt));
				else if (std::holds_alternative<bool>(opt))
					value = std::to_string(std::get<bool>(opt));
				syslog(LOG_INFO, "%s: %s", opt_pair.first.c_str(), value.c_str());
			}
			syslog(LOG_INFO, "Executing stamp");
		}

		// Make the stamp fail by default
		bool result = false;

		// Run the stamp code
		try
		{
			result = instance->run();
		}
		catch (std::exception &e)
		{
			syslog(LOG_ERR, "Internal error in rubberstamp: %s", e.what());
			continue;
		}

		if (verbose)
			syslog(LOG_INFO, "Stamp \"%s\" returned: %s", type.c_str(), std::to_string(result).c_str());

		// Abort authentication if the stamp returned false
		if (!result)
		{
			if (verbose)
				syslog(LOG_INFO, "Authentication aborted by rubber stamp");
			exit(14);
		}
	}

	// This is outside the for loop, so we've run all the rules
	if (verbose)
	{
		syslog(LOG_INFO, "All rubberstamps processed, authentication successful");
	}

	// Exit with no errors
	exit(0);
}
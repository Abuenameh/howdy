#ifndef RUBBER_STAMPS_H_
#define RUBBER_STAMPS_H_

#include <string>
#include <vector>
#include <variant>

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include <INIReader.h>

#include "video_capture.hh"
#include "models.hh"

struct OpenCV
{
	OpenCV(VideoCapture &video_capture, face_detection_model& face_detector, shape_predictor_model& pose_predictor, cv::Ptr<cv::CLAHE> clahe) : video_capture(video_capture), face_detector(face_detector), pose_predictor(pose_predictor), clahe(clahe)
	{}

	VideoCapture &video_capture;
	face_detection_model &face_detector;
	shape_predictor_model &pose_predictor;
	cv::Ptr<cv::CLAHE> clahe;
};

// /* Howdy rubber stamp */
// class RubberStamp
// {

// public:
// 	RubberStamp(INIReader &config, int gtk_proc, OpenCV &opencv);

// 	virtual ~RubberStamp() = default;

// 	/* Convert an ui string to input howdy-gtk understands */
// 	void set_ui_text(std::string text, TextType type = UI_TEXT);

// 	/* Write raw command to howdy-gtk stdin */
// 	void send_ui_raw(std::string command);

// 	virtual std::string name();

// 	virtual void declare_config();

// 	virtual bool run();

// 	bool verbose;
// 	INIReader &config;
// 	int gtk_proc;
// 	OpenCV &opencv;
// 	std::map<std::string, option> options;
// };

void execute(INIReader &config, int gtk_proc, OpenCV &opencv);

#endif // RUBBER_STAMPS_H_

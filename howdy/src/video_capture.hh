#ifndef VIDEO_CAPTURE_H_
#define VIDEO_CAPTURE_H_

#include <string>

#include <opencv2/videoio.hpp>

#include <INIReader.h>

class VideoCapture
{

public:
    /*
    Creates a new VideoCapture instance depending on the settings in the
    provided config file.
    */
    VideoCapture(INIReader& config_);

    /*
    Frees resources when destroyed
    */
    ~VideoCapture();

    /*
    Release cameras
    */
    void release();

    /*
    Reads a frame, returns the frame and an attempted grayscale conversion of
    the frame in a tuple:

    (frame, grayscale_frame)

    If the grayscale conversion fails, both items in the tuple are identical.
    */
    void read_frame(cv::Mat &frame, cv::Mat &gsframe);

    double get(int propId);

    bool set(int propId, double value);

    int fw;
    int fh;

private:
    INIReader& config;
    cv::VideoCapture internal;
};

#endif // VIDEO_CAPTURE_H_

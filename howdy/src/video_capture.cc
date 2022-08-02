#include <sys/syslog.h>
#include <syslog.h>

#include <filesystem>

#include <opencv2/imgproc.hpp>

#include "utils.hh"
#include "video_capture.hh"

namespace fs = std::filesystem;

/*
Creates a new VideoCapture instance depending on the settings in the
provided config file.
*/
VideoCapture::VideoCapture(INIReader config_) : config(config_)
{
    if (!fs::exists(fs::status(config.Get("video", "device_path", ""))))
    {
        if (config.GetBoolean("video", "warn_no_device", true))
        {
            syslog(LOG_ERR, "Howdy could not find a camera device at the path specified in the config file.");
            exit_code(1);
        }
    }

    // Create reader
    // The internal video recorder
    // Start video capture on the IR camera through OpenCV
    // internal = cv::VideoCapture(config.Get("video", "device_path", ""), cv::CAP_V4L);
    internal = cv::VideoCapture(config.Get("video", "device_path", ""));

    // Force MJPEG decoding if true
    if (config.GetBoolean("video", "force_mjpeg", false))
    {
        // Set a magic number, will enable MJPEG but is badly documentated
        internal.set(cv::CAP_PROP_FOURCC, 1196444237);
    }

    // Set the frame width and height if requested
    // The frame width
    fw = config.GetInteger("video", "frame_width", -1);
    // The frame height
    fh = config.GetInteger("video", "frame_height", -1);
    if (fw != -1)
        internal.set(cv::CAP_PROP_FRAME_WIDTH, fw);
    if (fh != -1)
        internal.set(cv::CAP_PROP_FRAME_HEIGHT, fh);

    // Request a frame to wake the camera up
    internal.grab();
}

/*
Frees resources when destroyed
*/
VideoCapture::~VideoCapture()
{
    internal.release();
}

/*
Release cameras
*/
void VideoCapture::release()
{
    internal.release();
}

/*
Reads a frame, returns the frame and an attempted grayscale conversion of
the frame in a tuple:

(frame, grayscale_frame)

If the grayscale conversion fails, both items in the tuple are identical.
*/
void VideoCapture::read_frame(cv::Mat &frame, cv::Mat &gsframe)
{
    bool ret = internal.read(frame);
    if (!ret)
    {
        syslog(LOG_ERR, "Failed to read camera specified in the 'device_path' config option, aborting");
        exit_code(1);
    }

    // Convert from color to grayscale
    cv::cvtColor(frame, gsframe, cv::COLOR_BGR2GRAY);
}

double VideoCapture::get(int propId)
{
    return internal.get(propId);
}

bool VideoCapture::set(int propId, double value)
{
    return internal.set(propId, value);
}

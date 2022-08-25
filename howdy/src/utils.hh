#ifndef UTILS_H_
#define UTILS_H_

#include <sys/syslog.h>
#include <syslog.h>

#include <chrono>

#include <opencv2/videoio.hpp>
#include <dlib/opencv.h>
#include <dlib/dnn.h>

using namespace dlib;

typedef std::chrono::time_point<std::chrono::system_clock> time_point;

const std::string PATH("/lib64/security/howdy");

inline void convert_image(cv::Mat &iimage, matrix<rgb_pixel> &oimage)
{
    if (iimage.channels() == 1)
    {
        assign_image(oimage, cv_image<unsigned char>(iimage));
    }
    else if (iimage.channels() == 3)
    {
        assign_image(oimage, cv_image<bgr_pixel>(iimage));
    }
    else
    {
        syslog(LOG_ERR, "Unsupported image type, must be 8bit gray or RGB image.");
        exit(1);
    }
}

inline time_point now()
{
    return std::chrono::system_clock::now();
}

#endif // UTILS_H_

#ifndef UTILS_H_
#define UTILS_H_

#include <chrono>

#include <opencv2/videoio.hpp>
#include <dlib/opencv.h>

using namespace dlib;

typedef std::chrono::time_point<std::chrono::system_clock> time_point;

const std::string PATH = "/lib64/security/howdy";

void exit_code(int code);

void convert_image(cv::Mat &iimage, matrix<rgb_pixel> &oimage);

time_point now();

#endif // UTILS_H_

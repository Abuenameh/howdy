#ifndef UTILS_H_
#define UTILS_H_

#include <opencv2/videoio.hpp>
#include <dlib/opencv.h>

using namespace dlib;

const std::string PATH = "/lib64/security/howdy";

void exit_code(int code);

void convert_image(cv::Mat &iimage, matrix<rgb_pixel> &oimage);

#endif // UTILS_H_

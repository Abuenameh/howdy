#ifndef SNAPSHOT_H_
#define SNAPSHOT_H_

#include <string>
#include <vector>

#include <opencv2/core.hpp>

std::string generate(std::vector<cv::Mat> &frames, std::vector<std::string> &text_lines);

#endif // SNAPSHOT_H_

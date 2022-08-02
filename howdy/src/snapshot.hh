#ifndef SNAPSHOT_H_
#define SNAPSHOT_H_

#include <string>
#include <vector>

#include <opencv2/core.hpp>

// class snapshot {
//     public:
//     snapshot(const std::string& path_);

// 	/* Generate a shapshot from given frames */
//     std::string generate(std::vector<cv::Mat>& frames, std::vector<std::string>& text_lines);

//     private:
//     const std::string PATH;
// };

    std::string generate(std::vector<cv::Mat>& frames, std::vector<std::string>& text_lines);

#endif // SNAPSHOT_H_

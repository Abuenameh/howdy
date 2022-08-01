#ifndef VIDEO_CAPTURE_H_
#define VIDEO_CAPTURE_H_

#include <string>

#include <INIReader.h>

class VideoCapture {

public:

    VideoCapture(std::string configFile);
    VideoCapture(INIReader contig);

    private:
    
}

#endif // VIDEO_CAPTURE_H_

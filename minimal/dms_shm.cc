#include <stdio.h>
#include <opencv2/opencv.hpp>


int main() {

    std::string rtsp1 = "rtmp://172.20.10.3:10035/live/sVNsJvqSR";

    cv::VideoCapture stream1 = cv::VideoCapture(rtsp1, cv::CAP_FFMPEG);

    if (!stream1.isOpened())
    {
        std::cout << "stream not opened." << std::endl;
        return -1;
    }

    cv::Mat frame1;

    while (true)
    {
        if (!stream1.read(frame1))
        {
            std::cout << "stream can not read." << std::endl;
            continue;
        }

        cv::imshow("input", frame1);

        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }

    return 0;
}

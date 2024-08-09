#include "InputFiles.hpp"
#include "opencv2/opencv.hpp"

int main() {
    const uint8_t* img = GetImgArray(0);
    cv::Mat src(128, 128, CV_8UC3, (void*)img);
    cv::Mat dst;
    cv::cvtColor(src, dst, cv::COLOR_RGB2BGR);
    cv::imshow("Image", dst);
    cv::waitKey(0);
    return 0;
}

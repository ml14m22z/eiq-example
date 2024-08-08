#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Please provide the path to the image as a command line argument!" << std::endl;
        return -1;
    }

    cv::Mat image = cv::imread(argv[1]);


    if (image.empty()) {
        std::cout << "Failed to load image!" << std::endl;
        return -1;
    }

    cv::imshow("Image", image);
    cv::waitKey(0);

    return 0;
}
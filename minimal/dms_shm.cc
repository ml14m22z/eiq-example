#include <stdio.h>
#include <opencv2/opencv.hpp>


std::vector<uint8_t> 
// cv::Mat
resizeCropImage(const cv::Mat& originalImage, const cv::Size& imageSize) {
    // IFM size
    int ifmWidth = imageSize.width;
    int ifmHeight = imageSize.height;

    // Aspect ratio resize
    float scaleRatio = static_cast<float>(std::max(ifmWidth, ifmHeight)) / 
                       static_cast<float>(std::min(originalImage.cols, originalImage.rows));
    int resizedWidth = static_cast<int>(originalImage.cols * scaleRatio);
    int resizedHeight = static_cast<int>(originalImage.rows * scaleRatio);
    cv::Mat resizedImage;
    cv::resize(originalImage, resizedImage, cv::Size(resizedWidth, resizedHeight), 0, 0, cv::INTER_LINEAR);

    // Crop the center of the image
    int left = (resizedWidth - ifmWidth) / 2;
    int top = (resizedHeight - ifmHeight) / 2;
    cv::Rect cropRegion(left, top, ifmWidth, ifmHeight);
    cv::Mat croppedImage = resizedImage(cropRegion);

    // return croppedImage;
    // Convert to a flattened array
    std::vector<uint8_t> flattenedImage;
    // flattenedImage.assign(croppedImage.datastart, croppedImage.dataend);
    flattenedImage.assign(croppedImage.data, croppedImage.data + croppedImage.total() * croppedImage.elemSize());
    
    return flattenedImage;
}


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

        cv::Size imageSize(128, 128);
        
        std::vector<uint8_t> data = resizeCropImage(frame1, imageSize);
        cv::Mat croppedImage = cv::Mat(imageSize, CV_8UC3, data.data());

        cv::imshow("cropped image", croppedImage);

        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }

    return 0;
}

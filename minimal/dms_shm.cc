#include <stdio.h>
#include <opencv2/opencv.hpp>


std::vector<uint8_t> 
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
    cv::Mat croppedImage = resizedImage(cropRegion).clone();

    // Convert to a flattened array
    std::vector<uint8_t> flattenedImage;
    // flattenedImage.assign(croppedImage.datastart, croppedImage.dataend);
    flattenedImage.assign(croppedImage.data, croppedImage.data + croppedImage.total() * croppedImage.elemSize());
    
    return flattenedImage;
}

void dumpData(const std::string& filename, const cv::Mat& image) {
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(image.data), image.total() * image.elemSize());
}

void dumpData(const std::string& filename, const std::vector<uint8_t>& data) {
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()), data.size());
}

// ping pong writer
void writeData(const std::string& filename, const std::vector<uint8_t>& data, int pingPong) {
    if (pingPong == 0) {
        std::string pingPongFilename = "ping_" + filename;
        dumpData(pingPongFilename, data);
    } else {
        std::string pingPongFilename = "pong_" + filename;
        dumpData(pingPongFilename, data);
    }
}


int main(int argc, char** argv) {

    cv::VideoCapture stream1 = cv::VideoCapture(argv[1], cv::CAP_FFMPEG);

    if (!stream1.isOpened())
    {
        std::cout << "stream not opened." << std::endl;
        return -1;
    }

    cv::Mat frame1;

    int pingPongReadyLast = 0;
    int pingPongReady = 0; // write when signal is changed

    while (true)
    {
        if (!stream1.read(frame1))
        {
            std::cout << "stream can not read." << std::endl;
            continue;
        }

        cv::Size imageSize(128, 128);
        
        std::vector<uint8_t> data = resizeCropImage(frame1, imageSize);
        cv::Mat img = cv::Mat(imageSize, CV_8UC3, data.data());
        
        // read ping pong ready signal from file pingPongReady.bin
        std::ifstream file("pingPongReady.bin", std::ios::binary);
        file.read(reinterpret_cast<char*>(&pingPongReady), sizeof(pingPongReady));
        file.close();

        if (pingPongReady != pingPongReadyLast) {
            pingPongReadyLast = pingPongReady;
            std::cout << "pingPongReady: " << pingPongReady << std::endl;
            writeData("img.bin", img, pingPongReady);
        }

        cv::imshow("img", img);

        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }

    return 0;
}

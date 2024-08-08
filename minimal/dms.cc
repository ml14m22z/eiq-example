#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/optional_debug_tools.h>

// using namespace std;
// using namespace cv;
// using namespace tflite;

const std::string MODEL_PATH = "../../models/";
const std::string DETECT_MODEL = "face_detection_front_128_full_integer_quant.tflite";
const std::string LANDMARK_MODEL = "face_landmark_192_integer_quant.tflite";
const std::string EYE_MODEL = "iris_landmark_quant.tflite";

const float SCORE_THRESH = 0.9;
const int MAX_FACE_NUM = 1;
const std::vector<int> ANCHOR_STRIDES = {8, 16};
const std::vector<int> ANCHOR_NUM = {2, 6};

Eigen::MatrixXf createAnchors(const cv::Size& inputShape) {
    int w = inputShape.width;
    int h = inputShape.height;
    Eigen::MatrixXf anchors;
    for (int i = 0; i < ANCHOR_STRIDES.size(); i++) {
        int s = ANCHOR_STRIDES[i];
        int aNum = ANCHOR_NUM[i];
        int gridCols = (w + s - 1) / s;
        int gridRows = (h + s - 1) / s;
        Eigen::VectorXf x(gridRows * gridCols * aNum);
        Eigen::VectorXf y(gridRows * gridCols * aNum);
        for (int r = 0; r < gridRows; r++) {
            for (int c = 0; c < gridCols; c++) {
                for (int a = 0; a < aNum; a++) {
                    int idx = r * gridCols * aNum + c * aNum + a;
                    x[idx] = (c + 0.5) * s;
                    y[idx] = (r + 0.5) * s;
                }
            }
        }
        // std::cout << "x.size: " << x.size() << std::endl;
        // std::cout << "y.size: " << y.size() << std::endl;
        
        // std::cout << "x.rows: " << x.rows() << std::endl;
        // std::cout << "x.cols: " << x.cols() << std::endl;
        // std::cout << "y.rows: " << y.rows() << std::endl;
        // std::cout << "y.cols: " << y.cols() << std::endl;

        Eigen::MatrixXf anchor_grid(gridRows * gridCols * aNum, 2);
        anchor_grid << y, x;
        // std::cout << "anchor_grid.size: " << anchor_grid.size() << std::endl;
        // std::cout << "anchor_grid.rows: " << anchor_grid.rows() << std::endl;
        // std::cout << "anchor_grid.cols: " << anchor_grid.cols() << std::endl;
        // std::cout << "anchor_grid: " << anchor_grid << std::endl;

        if (anchors.size() == 0) {
            anchors = anchor_grid;
        } else {
            anchors.conservativeResize(anchors.rows() + anchor_grid.rows(), Eigen::NoChange);
            anchors.bottomRows(anchor_grid.rows()) = anchor_grid;
        }
    }
    // std::cout << "anchors.size: " << anchors.size() << std::endl;
    return anchors;
}

// tuple<vector<Rect>, vector<vector<Point2f>>, vector<float>> decode(const vector<float>& scores, const vector<float>& bboxes, const Size& inputShape, const vector<vector<float>>& anchors) {
//     int w = inputShape.width;
//     int h = inputShape.height;
//     float topScore = *max_element(scores.begin(), scores.end());
//     float scoreThresh = max(SCORE_THRESH, topScore);
//     vector<Rect> predBbox;
//     vector<vector<Point2f>> landmarks;
//     vector<float> predScores;
//     for (int i = 0; i < scores.size(); i++) {
//         if (scores[i] >= scoreThresh) {
//             Rect bbox;
//             bbox.x = anchors[0][i] + bboxes[i * 4 + 1] * h;
//             bbox.y = anchors[1][i] + bboxes[i * 4] * w;
//             bbox.width = (anchors[0][i] + bboxes[i * 4 + 3] * h) - bbox.x;
//             bbox.height = (anchors[1][i] + bboxes[i * 4 + 2] * w) - bbox.y;
//             predBbox.push_back(bbox);

//             vector<Point2f> landmark;
//             for (int j = 0; j < 5; j++) {
//                 Point2f point;
//                 point.x = anchors[0][i] + bboxes[i * 10 + j * 2 + 5] * h;
//                 point.y = anchors[1][i] + bboxes[i * 10 + j * 2 + 4] * w;
//                 landmark.push_back(point);
//             }
//             landmarks.push_back(landmark);

//             predScores.push_back(scores[i]);
//         }
//     }
//     return make_tuple(predBbox, landmarks, predScores);
// }

// vector<int> nms(const vector<Rect>& bbox, const vector<float>& score, float thresh = 0.4) {
//     vector<int> keep;
//     vector<float> areas(bbox.size());
//     for (int i = 0; i < bbox.size(); i++) {
//         areas[i] = (bbox[i].width + 1) * (bbox[i].height + 1);
//     }
//     vector<int> order(score.size());
//     iota(order.begin(), order.end(), 0);
//     sort(order.begin(), order.end(), [&](int i, int j) { return score[i] > score[j]; });
//     while (!order.empty()) {
//         int i = order[0];
//         keep.push_back(i);
//         vector<int> inds;
//         for (int j = 1; j < order.size(); j++) {
//             int k = order[j];
//             int xx1 = max(bbox[i].x, bbox[k].x);
//             int yy1 = max(bbox[i].y, bbox[k].y);
//             int xx2 = min(bbox[i].x + bbox[i].width, bbox[k].x + bbox[k].width);
//             int yy2 = min(bbox[i].y + bbox[i].height, bbox[k].y + bbox[k].height);
//             int w = max(0, xx2 - xx1 + 1);
//             int h = max(0, yy2 - yy1 + 1);
//             float inter = w * h;
//             float ovr = inter / (areas[i] + areas[k] - inter);
//             if (ovr <= thresh) {
//                 inds.push_back(j);
//             }
//         }
//         vector<int> newOrder(inds.size());
//         for (int j = 0; j < inds.size(); j++) {
//             newOrder[j] = order[inds[j]];
//         }
//         order = newOrder;
//     }
//     return keep;
// }

cv::Mat resizeCropImage(const cv::Mat& originalImage, const cv::Size& imageSize) {
    int ifmWidth = imageSize.width;
    int ifmHeight = imageSize.height;
    float scaleRatio = std::max(float(ifmWidth) / std::min(originalImage.cols, originalImage.rows),
                           float(ifmHeight) / std::min(originalImage.cols, originalImage.rows));
    int resizedWidth = originalImage.cols * scaleRatio;
    int resizedHeight = originalImage.rows * scaleRatio;
    cv::Mat resizedImage;
    cv::resize(originalImage, resizedImage, cv::Size(resizedWidth, resizedHeight), 0, 0, cv::INTER_LINEAR);
    int left = (resizedWidth - ifmWidth) / 2;
    int top = (resizedHeight - ifmHeight) / 2;
    cv::Rect cropRect(left, top, ifmWidth, ifmHeight);
    cv::Mat croppedImage = resizedImage(cropRect);
    return croppedImage;
}

// Mat drawFaceBox(const Mat& image, const vector<Rect>& bboxes, const vector<vector<Point2f>>& landmarks, const vector<float>& scores) {
//     Mat result = image.clone();
//     for (int i = 0; i < bboxes.size(); i++) {
//         rectangle(result, bboxes[i], Scalar(255, 0, 0), 2);
//         for (int j = 0; j < landmarks[i].size(); j++) {
//             circle(result, landmarks[i][j], 2, Scalar(0, 255, 0), 2);
//             putText(result, to_string(j), landmarks[i][j] + Point2f(5, 5), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);
//         }
//         string scoreLabel = to_string(scores[i]);
//         Size labelSize = getTextSize(scoreLabel, FONT_HERSHEY_SIMPLEX, 1.0, 2, nullptr);
//         Point2f labelBottomLeft = bboxes[i].tl() + Point2f(10, labelSize.height + 10);
//         rectangle(result, bboxes[i].tl(), labelBottomLeft, Scalar(255, 0, 0), FILLED);
//         putText(result, scoreLabel, bboxes[i].tl() + Point2f(5, labelBottomLeft.y - 5), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);
//     }
//     return result;
// }

int main(int argc, char** argv) {
    std::string inputPath = "/dev/video0";
    if (argc > 1) {
        inputPath = argv[1];
    }

    // Load input image
    cv::Mat originalImage = cv::imread(inputPath);
    if (originalImage.empty()) {
        std::cerr << "Failed to load input image: " << inputPath << std::endl;
        return -1;
    }

    // Resize and crop input image
    cv::Size inputSize(128, 128);
    cv::Mat resizedImage = resizeCropImage(originalImage, inputSize);

    std::cout << "originalImage.size: " << originalImage.size() << std::endl;
    std::cout << "resizedImage.size: " << resizedImage.size() << std::endl;

    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile((MODEL_PATH + DETECT_MODEL).c_str());
    if (!model) {
        std::cerr << "Failed to load model: " << DETECT_MODEL << std::endl;
        return -1;
    }

    // Create an interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder builder(*model, resolver);
    if (builder(&interpreter) != kTfLiteOk) {
        fprintf(stderr, "Failed to build interpreter\n");
        return -1;
    }


    // Allocate tensor buffers
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        fprintf(stderr, "Failed to allocate tensors\n");
        return -1;
    }

    // Get input and output details
    int inputIndex = interpreter->inputs()[0];
    TfLiteIntArray* inputShape = interpreter->tensor(inputIndex)->dims;
    int outputClassificatorsIndex = interpreter->outputs()[0];
    int outputRegressorsIndex = interpreter->outputs()[1];

    // Create anchors
    Eigen::MatrixXf anchors = createAnchors(cv::Size(inputShape->data[2], inputShape->data[1]));

    // Show anchors
    // std::cout << anchors << std::endl;

    // Preprocess input data
    cv::Mat inputImage;
    resizedImage.convertTo(inputImage, CV_32F, 1.0 / 128.0, -1.0);
    std::cout << "inputImage.size: " << inputImage.size() << std::endl;
    inputImage = inputImage.reshape(1, inputImage.total());
    std::cout << "inputImage.size: " << inputImage.size() << std::endl;

    cv::imshow("resizedImage", resizedImage);
    cv::waitKey(0);

    // // Set input tensor
    // memcpy(interpreter->typed_tensor<float>(inputIndex), inputImage.data, inputImage.total() * sizeof(float));

    // // Run inference
    // interpreter->Invoke();

    // // Get output tensors
    // const TfLiteTensor* outputClassificators = interpreter->tensor(outputClassificatorsIndex);
    // const TfLiteTensor* outputRegressors = interpreter->tensor(outputRegressorsIndex);

    // // Decode output
    // vector<float> scores(outputClassificators->data.f, outputClassificators->data.f + outputClassificators->bytes / sizeof(float));
    // vector<float> bboxes(outputRegressors->data.f, outputRegressors->data.f + outputRegressors->bytes / sizeof(float));
    // vector<Rect> bboxesDecoded;
    // vector<vector<Point2f>> landmarks;
    // vector<float> predScores;
    // tie(bboxesDecoded, landmarks, predScores) = decode(scores, bboxes, inputSize, anchors);

    // // Apply NMS
    // vector<int> keepMask = nms(bboxesDecoded, predScores);
    // vector<Rect> bboxesFiltered;
    // vector<vector<Point2f>> landmarksFiltered;
    // vector<float> scoresFiltered;
    // for (int i = 0; i < keepMask.size(); i++) {
    //     bboxesFiltered.push_back(bboxesDecoded[keepMask[i]]);
    //     landmarksFiltered.push_back(landmarks[keepMask[i]]);
    //     scoresFiltered.push_back(predScores[keepMask[i]]);
    // }

    // // Draw face boxes and landmarks
    // Mat outputImage = drawFaceBox(originalImage, bboxesFiltered, landmarksFiltered, scoresFiltered);

    // // Show images
    // imshow("Input", originalImage);
    // imshow("Resized", resizedImage);
    // imshow("Output", outputImage);
    // waitKey(0);

    return 0;
}
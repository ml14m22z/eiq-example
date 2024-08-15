#include <iostream>
#include <string>
#include <sys/types.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include "InputFiles.hpp"

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

// auto 
std::tuple<std::vector<Eigen::Vector4d>, std::vector<std::vector<cv::Point2f>>, std::vector<float>>
decode(const std::vector<float>& scores, const std::vector<float>& bboxes, const cv::Size& inputShape, const Eigen::MatrixXf& anchors) {
    int w = inputShape.width;
    int h = inputShape.height;

    std::cout << "w: " << w << std::endl;
    std::cout << "h: " << h << std::endl;

    float topScore = *max_element(scores.begin(), scores.end());
    std::cout << "topScore: " << topScore << std::endl;

    float scoreThresh = std::max(SCORE_THRESH, topScore);
    std::cout << "scoreThresh: " << scoreThresh << std::endl;

    std::vector<Eigen::Vector4d> pred_bbox;
    std::vector<std::vector<cv::Point2f>> landmarks;
    std::vector<float> predScores;
    for (int i = 0; i < scores.size(); i++) {
        if (scores[i] >= scoreThresh) {
            std::cout << "scores[" << i << "]: " << scores[i] << std::endl;

            for (int j = 0; j < 16; j++) {
                std::cout << "bboxes[" << i << " * 16 + " << j << "]: " << bboxes[i * 16 + j] << std::endl;
            }

            std::cout << "anchors: " << anchors(i, 0) << " " << anchors(i, 1) << std::endl;

            float bboxes_decoded[2] = {(anchors(i, 0) + bboxes[i * 16 + 1]) / h, (anchors(i, 1) + bboxes[i * 16]) / w};
            std::cout << "bboxes_decoded: " << bboxes_decoded[0] << " " << bboxes_decoded[1] << std::endl;

            float pred_w = bboxes[i * 16 + 2] / w;
            float pred_h = bboxes[i * 16 + 3] / h;
            std::cout << "pred_w: " << pred_w << std::endl;
            std::cout << "pred_h: " << pred_h << std::endl;

            float topleft_x = bboxes_decoded[1] - pred_w * 0.5;
            float topleft_y = bboxes_decoded[0] - pred_h * 0.5;
            float btmright_x = bboxes_decoded[1] + pred_w * 0.5;
            float btmright_y = bboxes_decoded[0] + pred_h * 0.5;
            std::cout << "topleft_x: " << topleft_x << std::endl;
            std::cout << "topleft_y: " << topleft_y << std::endl;
            std::cout << "btmright_x: " << btmright_x << std::endl;
            std::cout << "btmright_y: " << btmright_y << std::endl;
            pred_bbox.push_back(Eigen::Vector4d(topleft_x, topleft_y, btmright_x, btmright_y));

            // cv::Rect bbox;
            // bbox.x = anchors(i, 0) + bboxes[i * 4 + 1] * h;
            // bbox.y = anchors(i, 1) + bboxes[i * 4] * w;
            // bbox.width = (anchors(i, 0) + bboxes[i * 4 + 3] * h) - bbox.x;
            // bbox.height = (anchors(i, 1) + bboxes[i * 4 + 2] * w) - bbox.y;
            // predBbox.push_back(bbox);
            // std::cout << "bbox: " << bbox << std::endl;

            // std::vector<cv::Point2f> landmark;
            // for (int j = 0; j < 5; j++) {
                // cv::Point2f point;
                // point.x = anchors(i, 0) + bboxes[i * 10 + j * 2 + 5] * h;
                // point.y = anchors(i, 1) + bboxes[i * 10 + j * 2 + 4] * w;
                // landmark.push_back(point);
                // std::cout << "landmark[" << j << "]: " << point << std::endl;
            // }
            // landmarks.push_back(landmark);

            std::vector<cv::Point2f> landmark;

            for (int j = 0; j < 6; j++) {
                cv::Point2f point;
                point.x = bboxes[i * 16 + j * 2 + 4];
                point.y = bboxes[i * 16 + j * 2 + 5];
                landmark.push_back(point);
            }

            std::cout << "landmark: " << landmark << std::endl;

            for (int j = 0; j < 6; j++) {
                landmark[j].x += anchors(i, 1);
                landmark[j].y += anchors(i, 0);
            }

            std::cout << "landmark: " << landmark << std::endl;

            for (int j = 0; j < 6; j++) {
                landmark[j].x /= h;
                landmark[j].y /= w;
            }

            std::cout << "landmark: " << landmark << std::endl;

            landmarks.push_back(landmark);

            predScores.push_back(scores[i]);
        }
    }
    return make_tuple(pred_bbox, landmarks, predScores);
}

std::vector<int> nms(const std::vector<Eigen::Vector4d>& bbox, const std::vector<float>& score, float thresh = 0.4) {
    std::vector<int> keep;
    std::vector<float> areas(bbox.size());
    for (int i = 0; i < bbox.size(); i++) {
        areas[i] = (bbox[i][2] - bbox[i][0] + 1) * (bbox[i][3] - bbox[i][1] + 1);
    }
    std::vector<int> order(score.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int i, int j) { return score[i] > score[j]; });
    while (!order.empty()) {
        int i = order[0];
        keep.push_back(i);
        std::vector<int> inds;
        for (int j = 1; j < order.size(); j++) {
            int k = order[j];
            int xx1 = std::max(bbox[i][0], bbox[k][0]);
            int yy1 = std::max(bbox[i][1], bbox[k][1]);
            int xx2 = std::min(bbox[i][2], bbox[k][2]);
            int yy2 = std::min(bbox[i][3], bbox[k][3]);
            int w = std::max(0, xx2 - xx1 + 1);
            int h = std::max(0, yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (areas[i] + areas[k] - inter);
            if (ovr <= thresh) {
                inds.push_back(j);
            }
        }
        std::vector<int> newOrder(inds.size());
        for (int j = 0; j < inds.size(); j++) {
            newOrder[j] = order[inds[j]];
        }
        order = newOrder;
    }
    return keep;
}

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

cv::Mat drawFaceBox(const cv::Mat& image, const std::vector<Eigen::Vector4d>& bboxes, const std::vector<std::vector<cv::Point2f>>& landmarks, const std::vector<float>& scores) {
    cv::Mat result = image.clone();
    for (int i = 0; i < bboxes.size(); i++) {
        cv::Rect bbox(bboxes[i][0], bboxes[i][1], bboxes[i][2] - bboxes[i][0], bboxes[i][3] - bboxes[i][1]);
        cv::rectangle(result, bbox, cv::Scalar(255, 0, 0), 2);
        // for (int j = 0; j < landmarks[i].size(); j++) {
        //     cv::circle(result, landmarks[i][j], 2, cv::Scalar(0, 255, 0), 2);
        //     cv::putText(result, std::to_string(j), landmarks[i][j] + cv::Point2f(5, 5), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
        // }
        std::string scoreLabel = std::to_string(scores[i]);
        if (scoreLabel.length() > 4) {
            scoreLabel = scoreLabel.substr(0, 4);
        }
        cv::Size labelSize = cv::getTextSize(scoreLabel, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, nullptr);
        cv::Point2f labelBottomRight = cv::Point2f(bbox.tl().x + labelSize.width + 10, bbox.tl().y + labelSize.height + 10);
        cv::rectangle(result, bbox.tl(), labelBottomRight, cv::Scalar(255, 0, 0), cv::FILLED);
        cv::putText(result, scoreLabel, cv::Point2f(bbox.tl().x + 5, labelBottomRight.y - 5), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
    }
    return result;
}

int dumpData(const float* data) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                std::cout << data[i * 3 * 128 + j * 3 + k] << " ";
            }
        }
        std::cout << std::endl;
    }
    return 0;
}

int dump(cv::Mat img) {
    std::cout << "img.size: " << img.size() << std::endl;
    std::cout << "img.channels: " << img.channels() << std::endl;
    std::cout << "img.depth: " << img.depth() << std::endl;
    std::cout << "img.type: " << cv::typeToString(img.type()) << std::endl;
    std::cout << "img: " << std::endl;
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            if (img.type() == CV_32FC3) {
                cv::Vec3f pixel = img.at<cv::Vec3f>(r, c);
                std::cout << pixel << " ";
            } else if (img.type() == CV_8UC3) {
                cv::Vec3b pixel = img.at<cv::Vec3b>(r, c);
                std::cout << pixel << " ";
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    return 0;
}

int main(int argc, char** argv) {

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
    cv::Mat rgbResizedImage(cv::Size(inputShape->data[2], inputShape->data[1]), CV_8UC3, (void*)GetImgArray(0));
    std::cout << "rgbResizedImage: " << std::endl;
    dump(rgbResizedImage);

    cv::Mat bgrResizedImage;
    cv::cvtColor(rgbResizedImage, bgrResizedImage, cv::COLOR_RGB2BGR);
    std::cout << "bgrResizedImage: " << std::endl;
    dump(bgrResizedImage);
    cv::imshow("bgrResizedImage", bgrResizedImage);

    cv::Mat inputImage;
    rgbResizedImage.convertTo(inputImage, CV_32FC3, 1.0 / 128.0, -1.0);
    // bgrResizedImage.convertTo(inputImage, CV_32F, 1.0 / 256.0, 0.0);
    std::cout << "inputImage: " << std::endl;
    dump(inputImage);
    cv::imshow("inputImage", inputImage);

    std::cout << "inputImage.size: " << inputImage.size() << std::endl;
    // inputImage = inputImage.reshape(3, inputImage.total());
    // std::cout << "inputImage.size: " << inputImage.size() << std::endl;

    std::cout << "inputImage: " << std::endl;
    dump(inputImage);
    
    std::cout << "inputImage.total: " << inputImage.total() << std::endl;
    std::cout << "sizeof(cv::Vec3f): " << sizeof(cv::Vec3f) << std::endl;

    // Set input tensor
    // memcpy(interpreter->typed_tensor<float>(inputIndex), inputImage.data, inputImage.total() * sizeof(float));
    memcpy(interpreter->typed_tensor<float>(inputIndex), inputImage.data, inputImage.total() * sizeof(cv::Vec3f));

    std::cout << "set input tensor done." << std::endl;
    // dump input tensor
    std::cout << "input tensor: " << std::endl;
    dumpData(interpreter->typed_tensor<float>(inputIndex));

    // Run inference
    interpreter->Invoke();

    // Get output tensors
    const TfLiteTensor* outputClassificators = interpreter->tensor(outputClassificatorsIndex);
    const TfLiteTensor* outputRegressors = interpreter->tensor(outputRegressorsIndex);

    // Decode output
    std::vector<float> scores(outputClassificators->data.f, outputClassificators->data.f + outputClassificators->bytes / sizeof(float));
    std::vector<float> bboxes(outputRegressors->data.f, outputRegressors->data.f + outputRegressors->bytes / sizeof(float));

    std::cout << "scores.size: " << scores.size() << std::endl;
    std::cout << "scores: " << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << scores[i] << " ";
    }
    std::cout << std::endl;

    // softmax: scores = 1 / (1 + np.exp(-scores))
    for (int i = 0; i < scores.size(); i++) {
        scores[i] = 1 / (1 + exp(-scores[i]));
    }

    std::cout << "bboxes.size: " << bboxes.size() << std::endl;
    std::cout << "bboxes: " << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << bboxes[i] << " ";
    }
    std::cout << std::endl;

    std::vector<Eigen::Vector4d> bboxesDecoded;
    std::vector<std::vector<cv::Point2f>> landmarks;
    std::vector<float> predScores;
    tie(bboxesDecoded, landmarks, predScores) = decode(scores, bboxes, cv::Size(inputShape->data[2], inputShape->data[1]), anchors);

    for (int i = 0; i < bboxesDecoded.size(); i++) {
        bboxesDecoded[i][0] *= rgbResizedImage.cols;
        bboxesDecoded[i][1] *= rgbResizedImage.rows;
        bboxesDecoded[i][2] *= rgbResizedImage.cols;
        bboxesDecoded[i][3] *= rgbResizedImage.rows;
    }

    for (int i = 0; i < landmarks.size(); i++) {
        for (int j = 0; j < landmarks[i].size(); j++) {
            landmarks[i][j].x *= rgbResizedImage.cols;
            landmarks[i][j].y *= rgbResizedImage.rows;
        }
    }

    std::cout << "bboxesDecoded.size: " << bboxesDecoded.size() << std::endl;
    std::cout << "bboxesDecoded: ";
    for (int i = 0; i < bboxesDecoded.size(); i++) {
        std::cout << bboxesDecoded[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "landmarks.size: " << landmarks.size() << std::endl;
    std::cout << "landmarks: ";
    for (int i = 0; i < landmarks.size(); i++) {
        std::cout << landmarks[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "predScores.size: " << predScores.size() << std::endl;
    std::cout << "predScores: ";
    for (int i = 0; i < predScores.size(); i++) {
        std::cout << predScores[i] << " ";
    }
    std::cout << std::endl;

    // Apply NMS
    std::vector<int> keepMask = nms(bboxesDecoded, predScores);

    std::cout << "keepMask.size: " << keepMask.size() << std::endl;
    std::cout << "keepMask: ";
    for (int i = 0; i < keepMask.size(); i++) {
        std::cout << keepMask[i] << " ";
    }
    std::cout << std::endl;

    std::vector<Eigen::Vector4d> bboxesFiltered;
    std::vector<std::vector<cv::Point2f>> landmarksFiltered;
    std::vector<float> scoresFiltered;
    for (int i = 0; i < keepMask.size(); i++) {
        bboxesFiltered.push_back(bboxesDecoded[keepMask[i]]);
        landmarksFiltered.push_back(landmarks[keepMask[i]]);
        scoresFiltered.push_back(predScores[keepMask[i]]);
    }

    std::cout << "bboxesFiltered.size: " << bboxesFiltered.size() << std::endl;
    std::cout << "bboxesFiltered: ";
    for (int i = 0; i < bboxesFiltered.size(); i++) {
        std::cout << bboxesFiltered[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "landmarksFiltered.size: " << landmarksFiltered.size() << std::endl;
    std::cout << "landmarksFiltered: ";
    for (int i = 0; i < landmarksFiltered.size(); i++) {
        std::cout << landmarksFiltered[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "scoresFiltered.size: " << scoresFiltered.size() << std::endl;
    std::cout << "scoresFiltered: ";
    for (int i = 0; i < scoresFiltered.size(); i++) {
        std::cout << scoresFiltered[i] << " ";
    }
    std::cout << std::endl;

    // Draw face boxes and landmarks
    cv::Mat outputImage = drawFaceBox(bgrResizedImage, bboxesFiltered, landmarksFiltered, scoresFiltered);

    // Show images
    cv::imshow("Output", outputImage);
    cv::waitKey(0);

    return 0;
}

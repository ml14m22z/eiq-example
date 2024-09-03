#include <iostream>
#include <string>
#include <sys/types.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
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

const std::vector<float> left_eye_pos = {0.38, 0.38};
const int target_width = 192;
const int target_height = target_width;

// Initialize the 3D face model
std::vector<cv::Point3f> FACE_MODEL_3D = {
    cv::Point3f(-165.0f / 4.5f, 170.0f  / 4.5f, -135.0f / 4.5f),    // left eye
    cv::Point3f(165.0f  / 4.5f, 170.0f  / 4.5f, -135.0f / 4.5f),    // right eye
    cv::Point3f(0.0f    / 4.5f, 0.0f    / 4.5f, 0.0f    / 4.5f),    // Nose
    cv::Point3f(0.0f    / 4.5f, -150.0f / 4.5f, -110.0f / 4.5f),    // mouth
    cv::Point3f(-330.0f / 4.5f, 100.0f  / 4.5f, -305.0f / 4.5f),    // left face
    cv::Point3f(330.0f  / 4.5f, 100.0f  / 4.5f, -305.0f / 4.5f)     // right face
};

float focal_length;
cv::Point2d camera_center;
cv::Mat camera_matrix;
cv::Mat dist_coeffs;

const int FACE_KEY_NUM = 468;

const int EYE_KEY_NUM = 71;
const int IRIS_KEY_NUM = 5;

int savetxt(const std::string& filename, const std::vector<auto>& data) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return -1;
    }
    for (const auto& d : data) {
        file << d << std::endl;
    }
    file.close();
    return 0;
}

int saveMat(const std::string& filename, const cv::Mat& data) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return -1;
    }
    for (int r = 0; r < data.rows; r++) {
        for (int c = 0; c < data.cols; c++) {
            // file << data.at<float>(r, c) << " ";
            file << data.at<cv::Vec3b>(r, c) << std::endl;
        }
        // file << std::endl;
    }
    file.close();
    return 0;
}

int saveMat3f(const std::string& filename, const cv::Mat& data) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return -1;
    }
    for (int r = 0; r < data.rows; r++) {
        for (int c = 0; c < data.cols; c++) {
            // file << data.at<float>(r, c) << " ";
            file << data.at<cv::Vec3f>(r, c) << std::endl;
        }
        // file << std::endl;
    }
    file.close();
    return 0;
}

cv::Mat padding(const cv::Mat& image) {
    int h = image.rows;
    int w = image.cols;
    int target_dim = std::max(w, h);
    int top = (target_dim - h) / 2;
    int bottom = (target_dim - h + 1) / 2;
    int left = (target_dim - w) / 2;
    int right = (target_dim - w + 1) / 2;

    cv::Mat padded;
    cv::copyMakeBorder(image, padded, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    // std::cout << "padded_size: [" << top << ", " << bottom << ", " << left << ", " << right << "]" << std::endl;

    return padded;
}

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
    // savetxt("scores.txt", scores);
    // savetxt("bboxes.txt", bboxes);
    
    int w = inputShape.width;
    int h = inputShape.height;

    // std::cout << "w: " << w << std::endl;
    // std::cout << "h: " << h << std::endl;

    float topScore = *max_element(scores.begin(), scores.end());
    // std::cout << "topScore: " << topScore << std::endl;

    float scoreThresh = std::max(SCORE_THRESH, topScore);
    // std::cout << "scoreThresh: " << scoreThresh << std::endl;

    std::vector<Eigen::Vector4d> pred_bbox;
    std::vector<std::vector<cv::Point2f>> landmarks;
    std::vector<float> predScores;
    for (int i = 0; i < scores.size(); i++) {
        if (scores[i] >= scoreThresh) {
            // std::cout << "scores[" << i << "]: " << scores[i] << std::endl;

            for (int j = 0; j < 16; j++) {
                // std::cout << "bboxes[" << i << " * 16 + " << j << "]: " << bboxes[i * 16 + j] << std::endl;
            }

            // std::cout << "anchors: " << anchors(i, 0) << " " << anchors(i, 1) << std::endl;

            float bboxes_decoded[2] = {(anchors(i, 0) + bboxes[i * 16 + 1]) / h, (anchors(i, 1) + bboxes[i * 16]) / w};
            // std::cout << "bboxes_decoded: " << bboxes_decoded[0] << " " << bboxes_decoded[1] << std::endl;

            float pred_w = bboxes[i * 16 + 2] / w;
            float pred_h = bboxes[i * 16 + 3] / h;
            // std::cout << "pred_w: " << pred_w << std::endl;
            // std::cout << "pred_h: " << pred_h << std::endl;

            float topleft_x = bboxes_decoded[1] - pred_w * 0.5;
            float topleft_y = bboxes_decoded[0] - pred_h * 0.5;
            float btmright_x = bboxes_decoded[1] + pred_w * 0.5;
            float btmright_y = bboxes_decoded[0] + pred_h * 0.5;
            // std::cout << "topleft_x: " << topleft_x << std::endl;
            // std::cout << "topleft_y: " << topleft_y << std::endl;
            // std::cout << "btmright_x: " << btmright_x << std::endl;
            // std::cout << "btmright_y: " << btmright_y << std::endl;
            pred_bbox.push_back(Eigen::Vector4d(topleft_x, topleft_y, btmright_x, btmright_y));
            // std::cout << "pred_bbox: " << pred_bbox.back() << std::endl;

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

            // std::cout << "landmark: " << landmark << std::endl;

            for (int j = 0; j < 6; j++) {
                landmark[j].x += anchors(i, 1);
                landmark[j].y += anchors(i, 0);
            }

            // std::cout << "landmark: " << landmark << std::endl;

            for (int j = 0; j < 6; j++) {
                landmark[j].x /= h;
                landmark[j].y /= w;
            }

            // std::cout << "landmark: " << landmark << std::endl;

            landmarks.push_back(landmark);

            predScores.push_back(scores[i]);
        }
    }
    return make_tuple(pred_bbox, landmarks, predScores);
}

// Function to get face angle
std::tuple<double, double, double> get_face_angle(const cv::Mat& rotation_vector, const cv::Mat& translation_vector) {
    cv::Mat rotation_mat;
    cv::Rodrigues(rotation_vector, rotation_mat);
    
    cv::Mat pose_mat;
    cv::hconcat(rotation_mat, translation_vector, pose_mat);

    cv::Mat euler_angle;
    cv::Mat cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ;
    cv::decomposeProjectionMatrix(pose_mat, cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, euler_angle);

    double pitch = euler_angle.at<double>(0) * M_PI / 180.0;
    double yaw = euler_angle.at<double>(1) * M_PI / 180.0;
    double roll = euler_angle.at<double>(2) * M_PI / 180.0;

    pitch = std::asin(std::sin(pitch)) * 180.0 / CV_PI;
    roll = -std::asin(std::sin(roll)) * 180.0 / CV_PI;
    yaw = std::asin(std::sin(yaw)) * 180.0 / CV_PI;

    return {pitch, roll, yaw};
}

// Function to get height-width ratio
double get_hw_ratio(const std::vector<cv::Point3f>& landmarks, const std::vector<int>& points) {
    int mouth_h = cv::norm(landmarks[points[0]] - landmarks[points[1]]);
    int mouth_w = cv::norm(landmarks[points[2]] - landmarks[points[3]]);
    return static_cast<double>(mouth_h) / mouth_w;
}

// Function to get mouth ratio
double get_mouth_ratio(const std::vector<cv::Point3f>& landmarks, const cv::Mat& image) {
    std::vector<int> POINTS = {13, 14, 78, 308}; // TOP, BOTTOM, LEFT, RIGHT
    return get_hw_ratio(landmarks, POINTS);
}

// Function to get eye ratio
double get_eye_ratio(const std::vector<cv::Point3f>& landmarks, const cv::Mat& image, const cv::Point& offsets = cv::Point(0, 0)) {
    std::vector<int> POINTS = {12, 4, 0, 8}; // TOP, BOTTOM, LEFT, RIGHT
    return get_hw_ratio(landmarks, POINTS);
}

// Function to get iris ratio
double get_iris_ratio(const std::vector<cv::Point3f>& left_landmarks, const std::vector<cv::Point3f>& right_landmarks) {
    int left = cv::norm(left_landmarks[1] - left_landmarks[3]);
    int right = cv::norm(right_landmarks[1] - right_landmarks[3]);
    return static_cast<double>(left) / right;
}

// Function to get eye boxes
std::tuple<cv::Rect, cv::Rect> get_eye_boxes(const std::vector<cv::Point3f>& landmarks, const cv::Size& size, double scale = 1.5) {
    auto get_box = [&](const std::vector<cv::Point3f>& landmarks, const std::vector<int>& points, double scale) {
        int x_min = size.width, y_min = size.height;
        int x_max = 0, y_max = 0;
        for (int i : points) {
            const auto& landmark = landmarks[i];
            x_min = std::min(static_cast<int>(landmark.x), x_min);
            y_min = std::min(static_cast<int>(landmark.y), y_min);
            x_max = std::max(static_cast<int>(landmark.x), x_max);
            y_max = std::max(static_cast<int>(landmark.y), y_max);
        }
        int x_mid = (x_max + x_min) / 2;
        int y_mid = (y_max + y_min) / 2;
        int box_len = static_cast<int>((x_max - x_min) * scale / 2);
        x_min = std::max(x_mid - box_len, 0);
        x_max = std::min(x_mid + box_len, size.width);
        y_min = std::max(y_mid - box_len, 0);
        y_max = std::min(y_mid + box_len, size.height);
        return cv::Rect(cv::Point(x_min, y_min), cv::Point(x_max, y_max));
    };

    std::vector<int> LEFT_EYE_POINT = {249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466};
    std::vector<int> RIGHT_EYE_POINT = {7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246};

    cv::Rect left_box = get_box(landmarks, LEFT_EYE_POINT, scale);
    cv::Rect right_box = get_box(landmarks, RIGHT_EYE_POINT, scale);
    return {left_box, right_box};
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

std::tuple<cv::Mat, cv::Mat, float> detect_align(const cv::Mat& image, const std::vector<cv::Point2f>& landmarks) {
    // get left and right eye
    cv::Point left_eye = landmarks[1];
    cv::Point right_eye = landmarks[0];
    // std::cout << "left_eye: " << left_eye << std::endl;
    // std::cout << "right_eye: " << right_eye << std::endl;

    // compute angle
    float dY = right_eye.y - left_eye.y;
    float dX = right_eye.x - left_eye.x;
    float angle = std::atan2(dY, dX) * 180.0 / CV_PI - 180.0;
    // std::cout << "dY: " << dY << std::endl;
    // std::cout << "dX: " << dX << std::endl;
    // std::cout << "angle: " << angle << std::endl;

    // compute the location of right/left eye in new image
    float right_eye_pos = 1.0 - left_eye_pos[0];
    // std::cout << "right_eye_pos: " << right_eye_pos << std::endl;

    // get the scale based on the distance
    float dist = std::sqrt(dY * dY + dX * dX);
    float desired_dist = (right_eye_pos - left_eye_pos[0]) * target_width;
    float scale = desired_dist / (dist + 1e-6);
    // std::cout << "dist: " << dist << std::endl;
    // std::cout << "desired_dist: " << desired_dist << std::endl;
    // std::cout << "scale: " << scale << std::endl;

    // get the center of eyes
    cv::Point2f eye_center((left_eye.x + right_eye.x) * 0.5, (left_eye.y + right_eye.y) * 0.5);
    // std::cout << "eye_center: " << eye_center << std::endl;

    // get transformation matrix
    cv::Mat M = cv::getRotationMatrix2D(eye_center, angle, scale);
    // std::cout << "M: " << M << std::endl;

    // align the center
    float tX = target_width * 0.5;
    float tY = target_height * left_eye_pos[1];
    M.at<double>(0, 2) += (tX - eye_center.x);
    M.at<double>(1, 2) += (tY - eye_center.y);  // update translation vector
    // std::cout << "tX: " << tX << std::endl;
    // std::cout << "tY: " << tY << std::endl;
    // std::cout << "M: " << M << std::endl;

    // apply affine transformation
    cv::Mat output;
    cv::warpAffine(image, output, M, cv::Size(target_width, target_height), cv::INTER_CUBIC);

    return std::make_tuple(output, M, angle);
}

// Method to invert the affine transformation matrix and apply it to the mesh landmarks
std::vector<cv::Point3f> detect_inverse(std::vector<cv::Point3f>& mesh_landmark, cv::Mat& M) {
    cv::Mat M_inverse;
    cv::invertAffineTransform(M, M_inverse);

    std::vector<cv::Point3f> mesh_landmark_inverse;
    for (const auto& point : mesh_landmark) {
        float px = M_inverse.at<double>(0, 0) * point.x + M_inverse.at<double>(0, 1) * point.y + M_inverse.at<double>(0, 2);
        float py = M_inverse.at<double>(1, 0) * point.x + M_inverse.at<double>(1, 1) * point.y + M_inverse.at<double>(1, 2);
        mesh_landmark_inverse.emplace_back(cv::Point3f(px, py, point.z));
    }

    return mesh_landmark_inverse;
}

// Method to solve the PnP problem and return the rotation and translation vectors
std::tuple<cv::Mat, cv::Mat> decode_pose(std::vector<cv::Point2f>& landmarks) {
    cv::Mat rotation_vector, translation_vector;

    // Assuming FACE_MODEL_3D, camera_matrix, and dist_coeffs are defined elsewhere
    cv::solvePnP(FACE_MODEL_3D, landmarks, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

    return std::make_tuple(rotation_vector, translation_vector);
}


int main(int argc, char** argv) {

    // Load model
    std::unique_ptr<tflite::FlatBufferModel> detect_model = tflite::FlatBufferModel::BuildFromFile((MODEL_PATH + DETECT_MODEL).c_str());
    if (!detect_model) {
        std::cerr << "Failed to load model: " << DETECT_MODEL << std::endl;
        return -1;
    }
    std::unique_ptr<tflite::FlatBufferModel> landmark_model = tflite::FlatBufferModel::BuildFromFile((MODEL_PATH + LANDMARK_MODEL).c_str());
    if (!detect_model) {
        std::cerr << "Failed to load model: " << LANDMARK_MODEL << std::endl;
        return -1;
    }
    std::unique_ptr<tflite::FlatBufferModel> eye_model = tflite::FlatBufferModel::BuildFromFile((MODEL_PATH + EYE_MODEL).c_str());
    if (!detect_model) {
        std::cerr << "Failed to load model: " << EYE_MODEL << std::endl;
        return -1;
    }

    // Create an interpreter
    tflite::ops::builtin::BuiltinOpResolver detect_resolver;
    std::unique_ptr<tflite::Interpreter> detect_interpreter;
    tflite::InterpreterBuilder detect_builder(*detect_model, detect_resolver);
    if (detect_builder(&detect_interpreter) != kTfLiteOk) {
        fprintf(stderr, "Failed to build detect_interpreter\n");
        return -1;
    }
    tflite::ops::builtin::BuiltinOpResolver landmark_resolver;
    std::unique_ptr<tflite::Interpreter> landmark_interpreter;
    tflite::InterpreterBuilder landmark_builder(*landmark_model, landmark_resolver);
    if (landmark_builder(&landmark_interpreter) != kTfLiteOk) {
        fprintf(stderr, "Failed to build landmark_interpreter\n");
        return -1;
    }
    tflite::ops::builtin::BuiltinOpResolver eye_resolver;
    std::unique_ptr<tflite::Interpreter> eye_interpreter;
    tflite::InterpreterBuilder eye_builder(*eye_model, eye_resolver);
    if (eye_builder(&eye_interpreter) != kTfLiteOk) {
        fprintf(stderr, "Failed to build eye_interpreter\n");
        return -1;
    }


    // Allocate tensor buffers
    if (detect_interpreter->AllocateTensors() != kTfLiteOk) {
        fprintf(stderr, "Failed to allocate detect tensors\n");
        return -1;
    }
    if (landmark_interpreter->AllocateTensors() != kTfLiteOk) {
        fprintf(stderr, "Failed to allocate landmark tensors\n");
        return -1;
    }
    if (eye_interpreter->AllocateTensors() != kTfLiteOk) {
        fprintf(stderr, "Failed to allocate eye tensors\n");
        return -1;
    }

    // Get input and output details
    int detect_inputIndex = detect_interpreter->inputs()[0];
    TfLiteIntArray* detect_inputShape = detect_interpreter->tensor(detect_inputIndex)->dims;
    int detect_outputClassificatorsIndex = detect_interpreter->outputs()[0];
    int detect_outputRegressorsIndex = detect_interpreter->outputs()[1];

    int landmark_inputIndex = landmark_interpreter->inputs()[0];
    TfLiteIntArray* landmark_inputShape = landmark_interpreter->tensor(landmark_inputIndex)->dims;
    int landmark_outputLandmarkIndex = landmark_interpreter->outputs()[0];
    int landmark_outputScoreIndex = landmark_interpreter->outputs()[1];

    int eye_inputIndex = eye_interpreter->inputs()[0];
    TfLiteIntArray* eye_inputShape = eye_interpreter->tensor(eye_inputIndex)->dims;
    int eye_outputIrisIndex = eye_interpreter->outputs()[0];
    int eye_outputEyeIndex = eye_interpreter->outputs()[1];

    // Create anchors
    Eigen::MatrixXf anchors = createAnchors(cv::Size(detect_inputShape->data[2], detect_inputShape->data[1]));

    // Show anchors
    // std::cout << anchors << std::endl;

    std::string rtsp1 = "rtmp://172.20.10.3:10035/live/sVNsJvqSR";
    cv::VideoCapture stream1 = cv::VideoCapture(rtsp1, cv::CAP_FFMPEG);
    stream1.set(CV_CAP_PROP_BUFFERSIZE, 0);

    if (!stream1.isOpened())
    {
        std::cout << "stream not opened" << std::endl;
        return -1;
    }

    cv::startWindowThread();
    cv::namedWindow("Input");
    cv::namedWindow("Output");

    // for (int imgIndex = 0; imgIndex < NUMBER_OF_FILES; imgIndex++) {
    for (;;) {

    // Preprocess input data
    // cv::Mat originalRgbImage(cv::Size(GetImgWidth(imgIndex), GetImgHeight(imgIndex)), CV_8UC3, (void*)GetImgArray(imgIndex));
    cv::Mat originalBgrImage;
    if (!stream1.read(originalBgrImage))
    {
        std::cout << "stream not read" << std::endl;
        continue;
    }
    cv::Mat originalRgbImage;
    cv::cvtColor(originalBgrImage, originalRgbImage, cv::COLOR_BGR2RGB);

    cv::Mat padded_rgb = padding(originalRgbImage);
    cv::Mat padded_bgr;
    cv::cvtColor(padded_rgb, padded_bgr, cv::COLOR_RGB2BGR);

    cv::Mat rgbResizedImage = resizeCropImage(padded_rgb, cv::Size(detect_inputShape->data[2], detect_inputShape->data[1]));

    cv::Mat bgrResizedImage;
    cv::cvtColor(rgbResizedImage, bgrResizedImage, cv::COLOR_RGB2BGR);

    cv::Mat detect_inputImageRgb;
    rgbResizedImage.convertTo(detect_inputImageRgb, CV_32FC3, 1.0 / 128.0, -1.0);

    cv::Mat detect_inputImageBgr;
    cv::cvtColor(detect_inputImageRgb, detect_inputImageBgr, cv::COLOR_RGB2BGR);

    // Set input tensor
    memcpy(detect_interpreter->typed_tensor<float>(detect_inputIndex), detect_inputImageRgb.data, detect_inputImageRgb.total() * sizeof(cv::Vec3f));

    // dump input tensor
    // std::cout << "input tensor: " << std::endl;
    // dumpData(detect_interpreter->typed_tensor<float>(detect_inputIndex));

    float* tensor_data = detect_interpreter->typed_tensor<float>(detect_inputIndex);
    int tensor_size = detect_interpreter->tensor(detect_inputIndex)->bytes / sizeof(float);
    std::vector<float> tensor_vector(tensor_data, tensor_data + tensor_size);
    // savetxt("input.txt", tensor_vector);

    // Run inference
    detect_interpreter->Invoke();

    // Get output tensors
    const TfLiteTensor* detect_outputClassificators = detect_interpreter->tensor(detect_outputClassificatorsIndex);
    const TfLiteTensor* detect_outputRegressors = detect_interpreter->tensor(detect_outputRegressorsIndex);

    // Decode output
    std::vector<float> scores(detect_outputClassificators->data.f, detect_outputClassificators->data.f + detect_outputClassificators->bytes / sizeof(float));
    std::vector<float> bboxes(detect_outputRegressors->data.f, detect_outputRegressors->data.f + detect_outputRegressors->bytes / sizeof(float));

    // std::cout << "scores.size: " << scores.size() << std::endl;
    // std::cout << "scores: " << std::endl;
    // for (int i = 0; i < 10; i++) {
    //     std::cout << scores[i] << " ";
    // }
    // std::cout << std::endl;

    // softmax: scores = 1 / (1 + np.exp(-scores))
    for (int i = 0; i < scores.size(); i++) {
        scores[i] = 1 / (1 + exp(-scores[i]));
    }

    // std::cout << "bboxes.size: " << bboxes.size() << std::endl;
    // std::cout << "bboxes: " << std::endl;
    // for (int i = 0; i < 10; i++) {
    //     std::cout << bboxes[i] << " ";
    // }
    // std::cout << std::endl;

    std::vector<Eigen::Vector4d> bboxesDecoded;
    std::vector<std::vector<cv::Point2f>> landmarks;
    std::vector<float> predScores;
    tie(bboxesDecoded, landmarks, predScores) = decode(scores, bboxes, cv::Size(detect_inputShape->data[2], detect_inputShape->data[1]), anchors);

    for (int i = 0; i < bboxesDecoded.size(); i++) {
        bboxesDecoded[i][0] *= padded_rgb.cols;
        bboxesDecoded[i][1] *= padded_rgb.rows;
        bboxesDecoded[i][2] *= padded_rgb.cols;
        bboxesDecoded[i][3] *= padded_rgb.rows;
    }

    for (int i = 0; i < landmarks.size(); i++) {
        for (int j = 0; j < landmarks[i].size(); j++) {
            landmarks[i][j].x *= padded_rgb.cols;
            landmarks[i][j].y *= padded_rgb.rows;
        }
    }

    // std::cout << "bboxesDecoded.size: " << bboxesDecoded.size() << std::endl;
    // std::cout << "bboxesDecoded: ";
    // for (int i = 0; i < bboxesDecoded.size(); i++) {
    //     std::cout << bboxesDecoded[i] << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "landmarks.size: " << landmarks.size() << std::endl;
    // std::cout << "landmarks: ";
    // for (int i = 0; i < landmarks.size(); i++) {
    //     std::cout << landmarks[i] << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "predScores.size: " << predScores.size() << std::endl;
    // std::cout << "predScores: ";
    // for (int i = 0; i < predScores.size(); i++) {
    //     std::cout << predScores[i] << " ";
    // }
    // std::cout << std::endl;

    // Apply NMS
    std::vector<int> keepMask = nms(bboxesDecoded, predScores);

    // std::cout << "keepMask.size: " << keepMask.size() << std::endl;
    // std::cout << "keepMask: ";
    // for (int i = 0; i < keepMask.size(); i++) {
    //     std::cout << keepMask[i] << " ";
    // }
    // std::cout << std::endl;

    std::vector<Eigen::Vector4d> bboxesFiltered;
    std::vector<std::vector<cv::Point2f>> landmarksFiltered;
    std::vector<float> scoresFiltered;
    for (int i = 0; i < keepMask.size(); i++) {
        bboxesFiltered.push_back(bboxesDecoded[keepMask[i]]);
        landmarksFiltered.push_back(landmarks[keepMask[i]]);
        scoresFiltered.push_back(predScores[keepMask[i]]);
    }

    // std::cout << "bboxesFiltered.size: " << bboxesFiltered.size() << std::endl;
    // std::cout << "bboxesFiltered: ";
    // for (int i = 0; i < bboxesFiltered.size(); i++) {
    //     std::cout << bboxesFiltered[i] << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "landmarksFiltered.size: " << landmarksFiltered.size() << std::endl;
    // std::cout << "landmarksFiltered: ";
    // for (int i = 0; i < landmarksFiltered.size(); i++) {
    //     std::cout << landmarksFiltered[i] << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "scoresFiltered.size: " << scoresFiltered.size() << std::endl;
    // std::cout << "scoresFiltered: ";
    // for (int i = 0; i < scoresFiltered.size(); i++) {
    //     std::cout << scoresFiltered[i] << " ";
    // }
    // std::cout << std::endl;

    const cv::Size& size = padded_rgb.size();
    // Initialize focal length and camera center
    focal_length = size.height;
    camera_center = cv::Point2d(size.width / 2.0, size.height / 2.0);
    // Initialize camera matrix
    camera_matrix = (cv::Mat_<float>(3, 3) << 
        focal_length, 0, camera_center.x,
        0, focal_length, camera_center.y,
        0, 0, 1);
    // Initialize distortion coefficients (assuming no lens distortion)
    dist_coeffs = cv::Mat::zeros(4, 1, CV_32F);

    std::vector<std::vector<cv::Point3f>> mesh_landmarks_inverse;
    std::vector<cv::Mat> r_vecs;
    std::vector<cv::Mat> t_vecs;

    // Loop through the bounding boxes and landmarks
    // std::cout << "bboxesDecoded.size: " << bboxesDecoded.size() << std::endl;
    for (size_t i = 0; i < bboxesDecoded.size(); ++i) {
        const auto& bbox = bboxesDecoded[i];
        std::vector<cv::Point2f> landmark = landmarks[i];
        // std::cout << "Bbox: " << bbox << std::endl;
        // std::cout << "Landmark: " << landmark << std::endl;

        // Align the face
        cv::Mat aligned_face, M;
        float angle;
        std::tie(aligned_face, M, angle) = detect_align(padded_rgb, landmark);
        // std::cout << "Aligned face: " << aligned_face << ", M: " << M << ", Angle: " << angle << std::endl;
        // std::cout << "Aligned face: " << std::endl;
        // dump(aligned_face);
        // std::cout << "M: " << M << ", Angle: " << angle << std::endl;

        // Perform mesh inference
        // std::vector<cv::Point3f> mesh_landmark;
        // std::vector<float> mesh_scores;
        // std::tie(mesh_landmark, mesh_scores) = landmark_inference(aligned_face);

        cv::Size landmark_size = cv::Size(landmark_inputShape->data[2], landmark_inputShape->data[1]);
        int h = landmark_size.height;
        int w = landmark_size.width;
        cv::Mat resized_aligned_face;
        cv::resize(aligned_face, resized_aligned_face, landmark_size, 0, 0, cv::INTER_LINEAR);
        cv::Mat landmark_inputImage;
        resized_aligned_face.convertTo(landmark_inputImage, CV_32FC3, 1.0 / 128.0, -1.0);

        // Set input tensor
        memcpy(landmark_interpreter->typed_tensor<float>(landmark_inputIndex), landmark_inputImage.data, landmark_inputImage.total() * sizeof(cv::Vec3f));

        // Run inference
        landmark_interpreter->Invoke();

        // Get output tensors
        const TfLiteTensor* landmark_outputLandmark = landmark_interpreter->tensor(landmark_outputLandmarkIndex);
        const TfLiteTensor* landmark_outputScore = landmark_interpreter->tensor(landmark_outputScoreIndex);

        // Decode output
        std::vector<float> landmarks_data(landmark_outputLandmark->data.f, landmark_outputLandmark->data.f + landmark_outputLandmark->bytes / sizeof(float));
        std::vector<float> scores_data(landmark_outputScore->data.f, landmark_outputScore->data.f + landmark_outputScore->bytes / sizeof(float));

        std::vector<cv::Point3f> mesh_landmark(FACE_KEY_NUM);
        for (int i = 0; i < FACE_KEY_NUM; ++i) {
            mesh_landmark[i].x = landmarks_data[i * 3 + 0] / w * resized_aligned_face.cols;
            mesh_landmark[i].y = landmarks_data[i * 3 + 1] / h * resized_aligned_face.rows;
            mesh_landmark[i].z = landmarks_data[i * 3 + 2];
        }

        float mesh_scores = 1.0f / (1.0f + std::exp(-scores_data[0]));

        // return std::make_tuple(mesh_landmark, mesh_scores);

        // Inverse the mesh landmarks
        std::vector<cv::Point3f> mesh_landmark_inverse = detect_inverse(mesh_landmark, M);
        mesh_landmarks_inverse.push_back(mesh_landmark_inverse);

        // Decode the pose
        cv::Mat r_vec, t_vec;
        std::tie(r_vec, t_vec) = decode_pose(landmark);
        r_vecs.push_back(r_vec);
        t_vecs.push_back(t_vec);
    }

    // Draw face boxes and landmarks
    cv::Mat outputImageRgb = drawFaceBox(padded_rgb, bboxesFiltered, landmarksFiltered, scoresFiltered);
    cv::Mat outputImageBgr;
    cv::cvtColor(outputImageRgb, outputImageBgr, cv::COLOR_RGB2BGR);

    for (size_t i = 0; i < mesh_landmarks_inverse.size(); i++) {
        const auto& mesh_landmark = mesh_landmarks_inverse[i];
        const auto& r_vec = r_vecs[i];
        const auto& t_vec = t_vecs[i];

        double mouth_ratio = get_mouth_ratio(mesh_landmark, outputImageRgb);
        auto [left_box, right_box] = get_eye_boxes(mesh_landmark, padded_rgb.size());

        cv::Mat left_eye_img_rgb = padded_rgb(left_box);
        cv::Mat left_eye_img_bgr;
        cv::cvtColor(left_eye_img_rgb, left_eye_img_bgr, cv::COLOR_RGB2BGR);

        cv::Mat right_eye_img_rgb = padded_rgb(right_box);
        cv::Mat right_eye_img_bgr;
        cv::cvtColor(right_eye_img_rgb, right_eye_img_bgr, cv::COLOR_RGB2BGR);


        // auto [left_eye_landmarks, left_iris_landmarks] = eye_mesher_inference(left_eye_img);
        cv::Mat left_eye_img_rgb_resized;
        cv::resize(left_eye_img_bgr, left_eye_img_rgb_resized, cv::Size(eye_inputShape->data[2], eye_inputShape->data[1]), 0, 0, cv::INTER_LINEAR);

        cv::Mat left_eye_inputImage;
        left_eye_img_rgb_resized.convertTo(left_eye_inputImage, CV_32FC3, 1.0 / 255.0);
        
        // Set input tensor
        memcpy(eye_interpreter->typed_tensor<float>(eye_inputIndex), left_eye_inputImage.data, left_eye_inputImage.total() * sizeof(cv::Vec3f));

        // Run inference
        eye_interpreter->Invoke();

        // Get output tensors
        const TfLiteTensor* left_eye_landmarks_data = eye_interpreter->tensor(eye_outputEyeIndex);
        const TfLiteTensor* left_iris_landmarks_data = eye_interpreter->tensor(eye_outputIrisIndex);

        // cv::Mat left_eye_landmarks(EYE_KEY_NUM, 3, CV_32F, left_eye_landmarks_data->data.f);
        // cv::Mat left_iris_landmarks(IRIS_KEY_NUM, 3, CV_32F, left_iris_landmarks_data->data.f);

        // for (int i = 0; i < EYE_KEY_NUM; ++i) {
        //     left_eye_landmarks.at<float>(i, 0) *= left_eye_img_rgb.cols / static_cast<float>(eye_inputShape->data[2]);
        //     left_eye_landmarks.at<float>(i, 1) *= left_eye_img_rgb.rows / static_cast<float>(eye_inputShape->data[1]);
        // }

        // for (int i = 0; i < IRIS_KEY_NUM; ++i) {
        //     left_iris_landmarks.at<float>(i, 0) *= left_eye_img_rgb.cols / static_cast<float>(eye_inputShape->data[2]);
        //     left_iris_landmarks.at<float>(i, 1) *= left_eye_img_rgb.rows / static_cast<float>(eye_inputShape->data[1]);
        // }

        std::vector<cv::Point3f> left_eye_landmarks(EYE_KEY_NUM);
        std::vector<cv::Point3f> left_iris_landmarks(IRIS_KEY_NUM);
        
        // Populate left_eye_landmarks
        for (int i = 0; i < EYE_KEY_NUM; ++i) {
            left_eye_landmarks[i] = cv::Point3f(
                left_eye_landmarks_data->data.f[i * 3],
                left_eye_landmarks_data->data.f[i * 3 + 1],
                left_eye_landmarks_data->data.f[i * 3 + 2]
            );
        }
        
        // Populate left_iris_landmarks
        for (int i = 0; i < IRIS_KEY_NUM; ++i) {
            left_iris_landmarks[i] = cv::Point3f(
                left_iris_landmarks_data->data.f[i * 3],
                left_iris_landmarks_data->data.f[i * 3 + 1],
                left_iris_landmarks_data->data.f[i * 3 + 2]
            );
        }
        
        for (int i = 0; i < EYE_KEY_NUM; ++i) {
            left_eye_landmarks[i].x *= left_eye_img_rgb.cols / static_cast<float>(eye_inputShape->data[2]);
            left_eye_landmarks[i].y *= left_eye_img_rgb.rows / static_cast<float>(eye_inputShape->data[1]);
        }
        
        for (int i = 0; i < IRIS_KEY_NUM; ++i) {
            left_iris_landmarks[i].x *= left_eye_img_rgb.cols / static_cast<float>(eye_inputShape->data[2]);
            left_iris_landmarks[i].y *= left_eye_img_rgb.rows / static_cast<float>(eye_inputShape->data[1]);
        }

        // auto [right_eye_landmarks, right_iris_landmarks] = eye_mesher_inference(right_eye_img);
        cv::Mat right_eye_img_rgb_resized;
        cv::resize(right_eye_img_rgb, right_eye_img_rgb_resized, cv::Size(eye_inputShape->data[2], eye_inputShape->data[1]), 0, 0, cv::INTER_LINEAR);
        cv::Mat right_eye_inputImage;
        right_eye_img_rgb_resized.convertTo(right_eye_inputImage, CV_32FC3, 1.0 / 255.0);
        
        // Set input tensor
        memcpy(eye_interpreter->typed_tensor<float>(eye_inputIndex), right_eye_inputImage.data, right_eye_inputImage.total() * sizeof(cv::Vec3f));

        // Run inference
        eye_interpreter->Invoke();

        // Get output tensors
        const TfLiteTensor* right_eye_landmarks_data = eye_interpreter->tensor(eye_outputEyeIndex);
        const TfLiteTensor* right_iris_landmarks_data = eye_interpreter->tensor(eye_outputIrisIndex);

        // cv::Mat right_eye_landmarks(EYE_KEY_NUM, 3, CV_32F, right_eye_landmarks_data->data.f);
        // cv::Mat right_iris_landmarks(IRIS_KEY_NUM, 3, CV_32F, right_iris_landmarks_data->data.f);

        // for (int i = 0; i < EYE_KEY_NUM; ++i) {
        //     right_eye_landmarks.at<float>(i, 0) *= right_eye_img_rgb.cols / static_cast<float>(eye_inputShape->data[2]);
        //     right_eye_landmarks.at<float>(i, 1) *= right_eye_img_rgb.rows / static_cast<float>(eye_inputShape->data[1]);
        // }

        // for (int i = 0; i < IRIS_KEY_NUM; ++i) {
        //     right_iris_landmarks.at<float>(i, 0) *= right_eye_img_rgb.cols / static_cast<float>(eye_inputShape->data[2]);
        //     right_iris_landmarks.at<float>(i, 1) *= right_eye_img_rgb.rows / static_cast<float>(eye_inputShape->data[1]);
        // }

        std::vector<cv::Point3f> right_eye_landmarks(EYE_KEY_NUM);
        std::vector<cv::Point3f> right_iris_landmarks(IRIS_KEY_NUM);

        // Populate right_eye_landmarks
        for (int i = 0; i < EYE_KEY_NUM; ++i) {
            right_eye_landmarks[i] = cv::Point3f(
                right_eye_landmarks_data->data.f[i * 3],
                right_eye_landmarks_data->data.f[i * 3 + 1],
                right_eye_landmarks_data->data.f[i * 3 + 2]
            );
        }

        // Populate right_iris_landmarks
        for (int i = 0; i < IRIS_KEY_NUM; ++i) {
            right_iris_landmarks[i] = cv::Point3f(
                right_iris_landmarks_data->data.f[i * 3],
                right_iris_landmarks_data->data.f[i * 3 + 1],
                right_iris_landmarks_data->data.f[i * 3 + 2]
            );
        }

        for (int i = 0; i < EYE_KEY_NUM; ++i) {
            right_eye_landmarks[i].x *= right_eye_img_rgb.cols / static_cast<float>(eye_inputShape->data[2]);
            right_eye_landmarks[i].y *= right_eye_img_rgb.rows / static_cast<float>(eye_inputShape->data[1]);
        }

        for (int i = 0; i < IRIS_KEY_NUM; ++i) {
            right_iris_landmarks[i].x *= right_eye_img_rgb.cols / static_cast<float>(eye_inputShape->data[2]);
            right_iris_landmarks[i].y *= right_eye_img_rgb.rows / static_cast<float>(eye_inputShape->data[1]);
        }


        double left_eye_ratio = get_eye_ratio(left_eye_landmarks, outputImageRgb, left_box.tl());
        double right_eye_ratio = get_eye_ratio(right_eye_landmarks, outputImageRgb, right_box.tl());
        auto [pitch, roll, yaw] = get_face_angle(r_vec, t_vec);
        double iris_ratio = get_iris_ratio(left_eye_landmarks, right_eye_landmarks);

        int h = originalRgbImage.rows;
        int w = originalRgbImage.cols;
        int target_dim = std::max(w, h);
        std::vector<int> padded_size = {(target_dim - h) / 2, (target_dim - h + 1) / 2,
                                        (target_dim - w) / 2, (target_dim - w + 1) / 2};

        if (mouth_ratio > 0.3) {
            cv::putText(outputImageRgb, "Yawning: Detected", cv::Point(padded_size[2] + 70, padded_size[0] + 70),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 0), 2);
        } else {
            cv::putText(outputImageRgb, "Yawning: No", cv::Point(padded_size[2] + 70, padded_size[0] + 70),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        }

        if (left_eye_ratio < 0.2 && right_eye_ratio < 0.2) {
            cv::putText(outputImageRgb, "Eye: Closed", cv::Point(padded_size[2] + 70, padded_size[0] + 100),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 0), 2);
        } else {
            cv::putText(outputImageRgb, "Eye: Open", cv::Point(padded_size[2] + 70, padded_size[0] + 100),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        }

        if (yaw > 15 && iris_ratio > 1.15) {
            cv::putText(outputImageRgb, "Face: Left", cv::Point(padded_size[2] + 70, padded_size[0] + 130),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 0), 2);
        } else if (yaw < -15 && iris_ratio < 0.85) {
            cv::putText(outputImageRgb, "Face: Right", cv::Point(padded_size[2] + 70, padded_size[0] + 130),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 0), 2);
        } else if (pitch > 30) {
            cv::putText(outputImageRgb, "Face: UP", cv::Point(padded_size[2] + 70, padded_size[0] + 130),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 0), 2);
        } else if (pitch < -13) {
            cv::putText(outputImageRgb, "Face: Down", cv::Point(padded_size[2] + 70, padded_size[0] + 130),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 0), 2);
        } else {
            cv::putText(outputImageRgb, "Face: Forward", cv::Point(padded_size[2] + 70, padded_size[0] + 130),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        }
    }
    cv::cvtColor(outputImageRgb, outputImageBgr, cv::COLOR_RGB2BGR);

    // remove pad
    int h = originalBgrImage.rows;
    int w = originalBgrImage.cols;
    int target_dim = std::max(w, h);
    int padded_size[] = {(target_dim - h) / 2, (target_dim - h + 1) / 2,
        (target_dim - w) / 2, (target_dim - w + 1) / 2};

    cv::Rect cropRect(padded_size[2], padded_size[0], w, h);
    cv::Mat outputImageRgbCropped = outputImageRgb(cropRect);
    cv::Mat outputImageBgrCropped;
    cv::cvtColor(outputImageRgbCropped, outputImageBgrCropped, cv::COLOR_RGB2BGR);

    // Show images
    // cv::imshow("Input", originalBgrImage);
    // cv::imshow("Output", outputImageBgrCropped);
    // cv::waitKey(1);

    cv::imshow("Input", originalBgrImage);
    cv::imshow("Output", outputImageBgrCropped);
    }

    return 0;
}

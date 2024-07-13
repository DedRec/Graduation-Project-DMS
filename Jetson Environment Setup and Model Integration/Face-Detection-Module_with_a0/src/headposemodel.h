#ifndef HEADPOSEMODEL_H
#define HEADPOSEMODEL_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include <cmath>

class HeadPoseModel {
public:
    HeadPoseModel(const std::string &modelPath);
    std::vector<float> predict(const cv::Mat &image);
    std::vector<float> computeEulerAnglesFromRotationMatrices(const std::vector<float> &rotationMatrices);

private:
    cv::dnn::Net net;
    cv::Size inputSize = cv::Size(224, 224);
    cv::Scalar mean = cv::Scalar(0.485, 0.456, 0.406);
    cv::Scalar std = cv::Scalar(0.229, 0.224, 0.225);
    
    cv::Mat preprocess(const cv::Mat &image);
};

#endif // HEADPOSEMODEL_H


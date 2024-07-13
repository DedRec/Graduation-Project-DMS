#include "headposemodel.h"

HeadPoseModel::HeadPoseModel(const std::string &modelPath) {
    net = cv::dnn::readNetFromONNX(modelPath);
net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);// CUDA , OPENCV
    	net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);// CUDA_FP16

    if (net.empty()) {
        throw std::runtime_error("Failed to load the model from: " + modelPath);
    }
}

cv::Mat HeadPoseModel::preprocess(const cv::Mat &image) {
    cv::Mat resizedImage, floatImage, normalizedImage;
    
    // Resize the image
    cv::resize(image, resizedImage, inputSize);
    
    // Convert the image to float
    resizedImage.convertTo(floatImage, CV_32F, 1.0 / 255.0);
    
    // Normalize the image
    cv::Mat meanMat = cv::Mat(floatImage.size(), CV_32FC3, mean);
    cv::Mat stdMat = cv::Mat(floatImage.size(), CV_32FC3, std);
    normalizedImage = (floatImage - meanMat) / stdMat;
    
    return normalizedImage;
}

std::vector<float> HeadPoseModel::predict(const cv::Mat &image) {
    cv::Mat preprocessedImage = preprocess(image);
    
    cv::Mat inputBlob = cv::dnn::blobFromImage(preprocessedImage);
    net.setInput(inputBlob);

    // Get output layer names
    std::vector<std::string> outputNames = net.getUnconnectedOutLayersNames();
    std::vector<cv::Mat> outputs;
    net.forward(outputs, outputNames);

    // Ensure the output dimensions are as expected
    if (outputs.size() != 3 || outputs[0].size[1] != 66 || outputs[1].size[1] != 66 || outputs[2].size[1] != 120) {
        std::cerr << "Unexpected output dimensions." << std::endl;
        return {};
    }
    
    // Find maximum positions in each output
    int maxPos1 = findMaxIndex(outputs[0]);
    int maxPos2 = findMaxIndex(outputs[1]);
    int maxPos3 = findMaxIndex(outputs[2]);

    // Map positions to degrees
    float degrees1 = mapPositionToDegrees(maxPos1, 0);
    float degrees2 = mapPositionToDegrees(maxPos2, 1);
    float degrees3 = mapPositionToDegrees(maxPos3, 2);

    // Return the degrees as a vector
    std::vector<float> degrees = {degrees1, degrees2, degrees3};
    return degrees;
}

float HeadPoseModel::findMaxValue(const cv::Mat &mat) {
    float maxValue = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < mat.total(); ++i) {
        float value = mat.at<float>(i);
        if (value > maxValue) {
            maxValue = value;
        }
    }
    return maxValue;
}

int HeadPoseModel::findMaxIndex(const cv::Mat &mat) {
    int maxIndex = 0;
    float maxValue = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < mat.total(); ++i) {
        float value = mat.at<float>(i);
        if (value > maxValue) {
            maxValue = value;
            maxIndex = i;
        }
    }
    return maxIndex;
}

float HeadPoseModel::mapPositionToDegrees(int maxPos, int vectorIndex) {
    float degrees = 0.0;
    if (vectorIndex == 0 || vectorIndex == 1) {
        // For vectors of size 66 (index 0 and 1)
        degrees = -99.0 + maxPos * 3.0;
    } else if (vectorIndex == 2) {
        // For vector of size 120 (index 2)
        degrees = -180.0 + maxPos * 3.0;
    }
    return degrees;
}


#include "eyegazemodel.h"

EyeGazeModel::EyeGazeModel(const std::string &modelPath) {
    net = cv::dnn::readNetFromONNX(modelPath);
net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);// CUDA , OPENCV
    	net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);// CUDA_FP16
    if (net.empty()) {
        throw std::runtime_error("Failed to load the model from: " + modelPath);
    }
}

cv::Mat EyeGazeModel::preprocess(const cv::Mat &image) {
    cv::Mat resizedImage, floatImage, normalizedImage;
    
    // Resize the image
    cv::resize(image, resizedImage, inputSize);
    
    // Convert the image to float
    resizedImage.convertTo(floatImage, CV_32F, 1.0 / 255.0);
    
    // Normalize the image
    cv::Mat meanMat = cv::Mat(floatImage.size(), CV_32FC3, mean);
    cv::Mat stdMat = cv::Mat(floatImage.size(), CV_32FC3, std);
    normalizedImage = (floatImage - meanMat) / stdMat;
    
    // Crop 54% of the height from the top
    //int height = normalizedImage.rows;
    //int cropped_height = int(0.54 * height);
    //int y1_cropped = 0;
    //int y2_cropped = cropped_height;
    //cv::Rect crop_region(0, y1_cropped, normalizedImage.cols, cropped_height);
    //cv::Mat croppedImage = normalizedImage(crop_region);
    
    return normalizedImage;
}

std::vector<float> EyeGazeModel::predict(const cv::Mat &image) {
    cv::Mat preprocessedImage = preprocess(image);
    
    cv::Mat inputBlob = cv::dnn::blobFromImage(preprocessedImage);
    net.setInput(inputBlob);
    cv::Mat output = net.forward();
    
    std::vector<float> predictions;
    for (int i = 0; i < output.size[1]; ++i) {
        predictions.push_back(output.at<float>(0, i));
    }
    
    return predictions;
}

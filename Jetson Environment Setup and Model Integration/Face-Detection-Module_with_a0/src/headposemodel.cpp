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

    // Change image from HWC to CHW
    //std::vector<cv::Mat> channels(3);
    //cv::split(normalizedImage, channels);
    //cv::Mat chwImage;
    //cv::vconcat(channels, chwImage);
    
    // Add a 4th dimension so that image new dimension is (1, C, H, W)
    //cv::Mat inputBlob = cv::dnn::blobFromImage(chwImage);
    cv::Mat inputBlob = cv::dnn::blobFromImage(normalizedImage);
    
    return inputBlob;
}

std::vector<float> HeadPoseModel::predict(const cv::Mat &image) {
    // Preprocess the input image
    cv::Mat preprocessedImage = preprocess(image);
    
    // Set the preprocessed image as the network input
    net.setInput(preprocessedImage);
    
    // Perform the forward pass to get the network output
    cv::Mat output = net.forward();
    
    // Convert the output to a std::vector<float>
    std::vector<float> predictions;
    /*for (int i = 0; i < output.cols; ++i) {
        predictions.push_back(output.at<float>(0, i));
    }*/
    
    
    for (int row = 0; row < 3; ++row) {
	    for (int col = 0; col < 3; ++col) {
		predictions.push_back(output.at<float>(row, col));
	    }
	}
    
    return predictions;
}

std::vector<float> HeadPoseModel::computeEulerAnglesFromRotationMatrices(const std::vector<float> &rotationMatrices) {
    std::vector<float> eulerAngles(3);

    // Convert the input vector to a 3x3 matrix
    cv::Mat R(3, 3, CV_32F);
    for (int i = 0; i < 9; ++i) {
        R.at<float>(i / 3, i % 3) = rotationMatrices[i];
    }

    float sy = std::sqrt(R.at<float>(0, 0) * R.at<float>(0, 0) + R.at<float>(1, 0) * R.at<float>(1, 0));

    bool singular = sy < 1e-6;

    float x, y, z;
    if (!singular) {
        x = std::atan2(R.at<float>(2, 1), R.at<float>(2, 2));
        y = std::atan2(-R.at<float>(2, 0), sy);
        z = std::atan2(R.at<float>(1, 0), R.at<float>(0, 0));
    } else {
        x = std::atan2(-R.at<float>(1, 2), R.at<float>(1, 1));
        y = std::atan2(-R.at<float>(2, 0), sy);
        z = 0;
    }

    //eulerAngles[0] = x;
    //eulerAngles[1] = y;
    //eulerAngles[2] = z;
// Convert radians to degrees
    eulerAngles[0] = x * (180.0 / M_PI);
    eulerAngles[1] = y * (180.0 / M_PI);
    eulerAngles[2] = z * (180.0 / M_PI);

    return eulerAngles;
}


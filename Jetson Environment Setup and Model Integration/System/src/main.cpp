#include "eyegazemodel.h"
#include "headposemodel.h"
#include "cv_dnn_ultraface.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

int main(int argc, char **argv) {
    std::string model_path = "../models/onnx/version-slim-320_simplified.onnx";
    std::string eye_model_path = "../models/eye_gaze/MNV3_small.onnx";
    std::string head_model_path = "../models/head_pose/runa0.onnx";
    
    UltraFace ultraface(model_path, 320, 240, 1, 0.7); // Config model input
    EyeGazeModel model(eye_model_path);
    HeadPoseModel headModel(head_model_path);
    
    // Open the camera
    // Read from video file
    std::string videofile = "../data/Test1.mp4";
    cv::VideoCapture cap(videofile);
    //cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the camera" << std::endl;
        return -1;
    }

    cv::namedWindow("UltraFace", cv::WINDOW_NORMAL); // Create a resizable window
    cv::resizeWindow("UltraFace", 224, 224); // Resize the window

    cv::Mat frame;
    std::vector<std::string> classes = {"Eyes Closed", "Forward", "Left Mirror", "Radio", "Rearview", "Right Mirror", "Shoulder", "Speedometer"};

	// Variables to track FPS
    double minFPS = std::numeric_limits<double>::max();
    double maxFPS = 0.0;
    std::vector<double> fpsValues;    

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Empty frame grabbed" << std::endl;
            break; // Skip this frame
        }
	
	
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            std::vector<FaceInfo> face_list;
            ultraface.detect(frame, face_list);

            float maxConf = 0;
            cv::Rect bestFaceRect;
            for (const auto& face : face_list) {
                if (face.score > maxConf) {
                    maxConf = face.score;
                    bestFaceRect = cv::Rect(face.x1, face.y1, face.x2 - face.x1, face.y2 - face.y1);
                }
            }

	if (bestFaceRect.empty()) {
	    continue;
	} 
            
            // Crop the head region
            cv::Mat headROI = frame(bestFaceRect);
            
            // Perform prediction on the head region
            std::vector<float> anglePredictions = headModel.predict(headROI);
            
            // Crop 54% of the height from the top of the bounding box
            int cropped_height = int(0.54 * bestFaceRect.height);
            cv::Rect crop_region(bestFaceRect.x, bestFaceRect.y, bestFaceRect.width, cropped_height);
            cv::rectangle(frame, crop_region, cv::Scalar(0, 255, 0), 2);

            // Crop the face region
            cv::Mat faceROI = frame(crop_region);

            // Perform prediction on the cropped face region
            std::vector<float> predictions = model.predict(faceROI);
            
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            //std::cout << "FPS: " << 1.0 / elapsed.count() << std::endl;

	// Calculate FPS
        double fps = 1.0 / elapsed.count();
std::cout << "FPS: " << fps << std::endl;
        fpsValues.push_back(fps);
// Update min and max FPS
        if (fps < minFPS) minFPS = fps;
        if (fps > maxFPS) maxFPS = fps;

// Calculate average FPS
    double sumFPS = 0.0;
    for (double fps : fpsValues) {
        sumFPS += fps;
    }
    double avgFPS = sumFPS / fpsValues.size();

// Output FPS statistics
    std::cout << "Min FPS: " << minFPS << std::endl;
    std::cout << "Max FPS: " << maxFPS << std::endl;
    std::cout << "Avg FPS: " << avgFPS << std::endl;       

            std::cout << "Angle Predictions: ";
	    for (float angle : anglePredictions) {
		std::cout << angle << " ";
	    }
	    std::cout << std::endl;

	// Compute Euler angles
            //std::vector<float> eulerAngles = headModel.computeEulerAnglesFromRotationMatrices(anglePredictions);

            // Output the Euler angles
            //std::cout << "Yaw: " << eulerAngles[0] << ", Pitch: " << eulerAngles[1] << ", Roll: " << eulerAngles[2] << std::endl;
	    
	    //Draw BBox on Face
            //cv::rectangle(frame, bestFaceRect, cv::Scalar(255, 0, 0), 2);
            
            // Find the class with the maximum prediction
            auto maxIt = std::max_element(predictions.begin(), predictions.end());
            int maxIndex = std::distance(predictions.begin(), maxIt);
            std::string maxClass = classes[maxIndex];
            float maxPrediction = *maxIt;

            // Display the class with the maximum prediction on the frame
            std::string text = maxClass + ": " + std::to_string(maxPrediction);
            int fontFace = cv::FONT_HERSHEY_SIMPLEX;
            double fontScale = 0.7;
            int thickness = 2;
            cv::Point textOrg(crop_region.x, crop_region.y - 10);
            cv::putText(frame, text, textOrg, fontFace, fontScale, cv::Scalar(0, 255, 0), thickness);
// Format FPS and max FPS to 2 decimal points
            std::ostringstream fpsStream, maxFpsStream;
            fpsStream << std::fixed << std::setprecision(2) << fps;
            maxFpsStream << std::fixed << std::setprecision(2) << maxFPS;
            std::string fpsText = "FPS: " + fpsStream.str();
            std::string maxFpsText = "Max FPS: " + maxFpsStream.str();
            
            // Display the FPS and max FPS on the frame with smaller font size
            double smallFontScale = 0.5;
            int smallFontThickness = 1;
            cv::putText(frame, fpsText, cv::Point(10, 30), fontFace, smallFontScale, cv::Scalar(0, 255, 0), smallFontThickness);
            cv::putText(frame, maxFpsText, cv::Point(10, 50), fontFace, smallFontScale, cv::Scalar(0, 255, 0), smallFontThickness);

        } catch (const cv::Exception& e) {
            std::cerr << "OpenCV error: " << e.what() << std::endl;
        }

        cv::imshow("UltraFace", frame);
        if (cv::waitKey(1) == 27) { // Press 'Esc' to exit
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}

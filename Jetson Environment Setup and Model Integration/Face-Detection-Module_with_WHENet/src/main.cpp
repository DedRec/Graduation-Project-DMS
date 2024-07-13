#include "headposemodel.h"
#include "cv_dnn_ultraface.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

int main(int argc, char **argv) {
    std::string model_path = "../models/onnx/version-slim-320_simplified.onnx";
    std::string head_model_path = "../models/head_pose/model_MNV3_WHENet.onnx";
    
    UltraFace ultraface(model_path, 320, 240, 1, 0.7); // Config model input
    HeadPoseModel headModel(head_model_path);
    
    // Read from video file
    std::string videofile = "../data/Test1.mp4";
cv::VideoCapture cap(videofile);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the camera" << std::endl;
        return -1;
    }

    cv::namedWindow("UltraFace", cv::WINDOW_NORMAL); // Create a resizable window
    cv::resizeWindow("UltraFace", 224, 224); // Resize the window

    cv::Mat frame;

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
        auto end = std::chrono::high_resolution_clock::now();
        
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
            
            end = std::chrono::high_resolution_clock::now();
            
		size_t vectorSize = anglePredictions.size();
            std::cout << "Angle Predictions: ";
	    if (vectorSize == 3) {
		    std::cout << "Roll: " << anglePredictions[0] << std::endl;
		    std::cout << "Pitch: " << anglePredictions[1] << std::endl;
		    std::cout << "Yaw: " << anglePredictions[2] << std::endl;
		} else {
		    std::cout << "Unexpected size of anglePredictions" << std::endl;
		}
	    std::cout << std::endl;
	    
	    //Draw BBox on Face
            cv::rectangle(frame, bestFaceRect, cv::Scalar(255, 0, 0), 4);
            
        } catch (const cv::Exception& e) {
            std::cerr << "OpenCV error: " << e.what() << std::endl;
        }

        std::chrono::duration<double> elapsed = end - start;
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

        cv::imshow("UltraFace", frame);
        if (cv::waitKey(1) == 27) { // Press 'Esc' to exit
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}


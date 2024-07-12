#include "eyegazemodel.h"
#include "headposemodel.h"
#include "cv_dnn_ultraface.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

int main(int argc, char **argv) {
    std::string model_path = "../models/onnx/version-slim-320_simplified.onnx";
    std::string eye_model_path = "../models/eye_gaze/MNV3_small.onnx";
    std::string head_model_path = "../models/head_pose/Trainon300w-lpTestonBIWIbackboneRepVGG-A0_epoch_80.onnx";
    
    UltraFace ultraface(model_path, 320, 240, 1, 0.7); // Config model input
    EyeGazeModel model(eye_model_path);
    HeadPoseModel headModel(head_model_path);
    
    // Open the camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the camera" << std::endl;
        return -1;
    }

    cv::namedWindow("UltraFace", cv::WINDOW_NORMAL); // Create a resizable window
    cv::resizeWindow("UltraFace", 224, 224); // Resize the window
    
    //cv::namedWindow("UltraFaceHPE", cv::WINDOW_NORMAL); // Create a resizable window
    //cv::resizeWindow("UltraFaceHPE", 224, 224); // Resize the window

    cv::Mat frame;
    cv::Mat frame1;
    std::vector<std::string> classes = {"Eyes Closed", "Forward", "Left Mirror", "Radio", "Rearview", "Right Mirror", "Shoulder", "Speedometer"};
    
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
            
            //frame1 = frame.clone();
            
            // Crop the head region
            cv::Mat headROI = frame(bestFaceRect);
            
            // Perform prediction on the head region
            std::vector<float> anglePredictions = headModel.predict(headROI);
            
            // Crop 54% of the height from the top of the bounding box
            // int height = bestFaceRect.height;
            int cropped_height = int(0.54 * bestFaceRect.height);
            cv::Rect crop_region(bestFaceRect.x, bestFaceRect.y, bestFaceRect.width, cropped_height);
            cv::rectangle(frame, crop_region, cv::Scalar(0, 255, 0), 2);

            // Crop the face region
            cv::Mat faceROI = frame(crop_region);

            // Perform prediction on the cropped face region
            std::vector<float> predictions = model.predict(faceROI);
            
            end = std::chrono::high_resolution_clock::now();
            
            std::cout << "Angle Predictions: ";
	    for (float angle : anglePredictions) {
		std::cout << angle << " ";
	    }
	    std::cout << std::endl;
	    
	    //Draw BBox on Face
            cv::rectangle(frame, bestFaceRect, cv::Scalar(255, 0, 0), 4);
            
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
            
            // Convert float angles to string for display
		/*std::stringstream ss;
		ss << "Yaw: " << anglePredictions[0] << ", Pitch: " << anglePredictions[1] << ", Roll: " << anglePredictions[2];
		std::string headtext = ss.str();

		// Display the text on frame1
		cv::Point headtextOrg(bestFaceRect.x, bestFaceRect.y - 20);  // Adjust position as needed
		cv::putText(frame, headtext, headtextOrg, fontFace, fontScale, cv::Scalar(0, 255, 0), thickness);*/

        } catch (const cv::Exception& e) {
            std::cerr << "OpenCV error: " << e.what() << std::endl;
        }

        //auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "FPS: " << 1.0 / elapsed.count() << std::endl;

        cv::imshow("UltraFace", frame);
        //cv::imshow("UltraFaceHPE", frame1);
        if (cv::waitKey(1) == 27) { // Press 'Esc' to exit
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}


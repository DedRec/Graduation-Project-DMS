#include "headposemodel.h"
#include "cv_dnn_ultraface.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

int main(int argc, char **argv) {
    std::string model_path = "../models/onnx/version-slim-320_simplified.onnx";
    std::string head_model_path = "../models/head_pose/Trainon300w-lpTestonBIWIbackboneRepVGG-A0_epoch_80.onnx";
    
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
            
            // Crop the head region
            cv::Mat headROI = frame(bestFaceRect);
            
            // Perform prediction on the head region
            std::vector<float> anglePredictions = headModel.predict(headROI);
            
            end = std::chrono::high_resolution_clock::now();
            
            std::cout << "Angle Predictions: ";
	    for (float angle : anglePredictions) {
		std::cout << angle << " ";
	    }
	    std::cout << std::endl;
	    
	    //Draw BBox on Face
            cv::rectangle(frame, bestFaceRect, cv::Scalar(255, 0, 0), 4);
            
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


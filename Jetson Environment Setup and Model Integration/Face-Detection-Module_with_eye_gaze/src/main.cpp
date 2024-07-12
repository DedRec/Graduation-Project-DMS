#include "eyegazemodel.h"
#include "cv_dnn_ultraface.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

int main(int argc, char **argv) {
    std::string model_path = "../models/onnx/version-slim-320_simplified.onnx";
    std::string eye_model_path = "../models/eye_gaze/MNV3_small.onnx";
    UltraFace ultraface(model_path, 320, 240, 1, 0.7); // Config model input
    EyeGazeModel model(eye_model_path);
    
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
            
            // Crop 54% of the height from the top of the bounding box
            int height = bestFaceRect.height;
            int cropped_height = int(0.54 * height);
            cv::Rect crop_region(bestFaceRect.x, bestFaceRect.y, bestFaceRect.width, cropped_height);
            cv::rectangle(frame, crop_region, cv::Scalar(0, 255, 0), 2);

            // Crop the face region
            cv::Mat faceROI = frame(crop_region);

            // Perform prediction on the cropped face region
            std::vector<float> predictions = model.predict(faceROI);
            
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

        } catch (const cv::Exception& e) {
            std::cerr << "OpenCV error: " << e.what() << std::endl;
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "FPS: " << 1.0 / elapsed.count() << std::endl;

        cv::imshow("UltraFace", frame);
        if (cv::waitKey(1) == 27) { // Press 'Esc' to exit
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}


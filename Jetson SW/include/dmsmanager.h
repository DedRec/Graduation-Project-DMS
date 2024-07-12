#pragma once

#include <thread>
#include <signal.h>
#include "threadsafequeue.h"
#include "basiccameracomponent.h"
#include "facedetectioncomponent.h"
#include "aicomponent.h"
#include "commtcpcomponent.h"

#define COMMANDS_NUM        10
typedef unsigned char  Std_ReturnType;
#define E_OK            ((Std_ReturnType)0x00U)      /* Function Return OK */
#define E_NOT_OK        ((Std_ReturnType)0x01U)      /* Function Return NOT OK */

#define TURN_SYSTEM_ON      (0x1)
#define TURN_SYSTEM_OFF     (0x2)

typedef enum
{
    SET_FPS,
    SET_FDT,
    TURN_SYSTEM,
    SET_FD_MODEL,
    SET_HP_MODEL,
    SET_EG_MODEL,
}COMMANDS;

typedef enum
{
    AX,
    AY,
    AZ,
    A0,
    eff0,
    eff1,
    eff2,
    eff3,
    whenNet,
    NoHeadPose
}HEADPOSE_MODELS;

typedef enum
{
    mobilenetv3,
    squeezenet,
    resnet,
    mobilenet,
    NoEyeGaze
}EYEGAZE_MODELS;

typedef enum
{
    YoloV3Tiny,
    YoloV2,
    NoFaceDetection
}FACEDETECTION_MODELS;


class DMSManager {
public:
    DMSManager(ThreadSafeQueue<cv::Mat>& cameraQueue,
               ThreadSafeQueue<cv::Mat>& faceDetectionQueue, ThreadSafeQueue<cv::Rect>& faceRectQueue,
               ThreadSafeQueue<std::vector<std::vector<float>>>& AIDetectionQueue, 
               ThreadSafeQueue<cv::Mat>& framesQueue, 
               ThreadSafeQueue<cv::Mat>& tcpOutputQueue, 
               int tcpPort,
               ThreadSafeQueue<std::string>& commandsQueue, ThreadSafeQueue<std::string>& faultsQueue);
    ~DMSManager();

    bool startSystem();
    void stopSystem();
    bool initializeCamera(const std::string& source);
    void setCameraFPS(int fps);
    void setFaceFDT(int fdt);
    void setCamereSource(const std::string& source);
    void clearQueues();
    void setupSignalHandlers();

    Std_ReturnType setFPS(unsigned short command);
    Std_ReturnType setFDT(unsigned short command);
    Std_ReturnType turnSystem(unsigned short command);
    Std_ReturnType setFDmodel(unsigned short command);
    Std_ReturnType setHPmodel(unsigned short command);
    Std_ReturnType setEGmodel(unsigned short command);
    Std_ReturnType setSource(const std::string& command);

    // Function to handle the different types of commands 
    void handleCommand(std::string& command);

private:
    BasicCameraComponent cameraComponent;
    FaceDetectionComponent faceDetectionComponent;
    AIComponent AiComponent;
    CommTCPComponent tcpComponent; 

    ThreadSafeQueue<cv::Mat>& cameraQueue;
    ThreadSafeQueue<cv::Mat>& faceDetectionQueue;
    ThreadSafeQueue<cv::Rect>& faceRectQueue;
    ThreadSafeQueue<std::vector<std::vector<float>>>& AIDetectionQueue;
    ThreadSafeQueue<cv::Mat>& framesQueue;
    ThreadSafeQueue<cv::Mat>& tcpOutputQueue;
    ThreadSafeQueue<std::string>& commandsQueue;
    ThreadSafeQueue<std::string>& faultsQueue;

    std::thread cameraThread;
    std::thread faceDetectionThread;
    std::thread AIThread;
    std::thread tcpThread; 
    std::thread commandsThread; 

    int tcpPort; 
    bool running;
    bool firstRun = true;

    // Component loops that start in their own thread
    void cameraLoop();
    void faceDetectionLoop();
    void AILoop();
    void commtcpLoop(); 
    void commandsLoop();
};


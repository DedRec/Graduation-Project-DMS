#include "dmsmanager.h"
#include <benchmark/benchmark.h>

Std_ReturnType (*func_ptrs[COMMANDS_NUM])(unsigned short)={NULL};

// Constructor: passes input and output queues for different components
DMSManager::DMSManager(ThreadSafeQueue<cv::Mat>& cameraQueue, 
                       ThreadSafeQueue<cv::Mat>& faceDetectionQueue, 
                       ThreadSafeQueue<cv::Rect>& faceRectQueue,
                       ThreadSafeQueue<std::vector<std::vector<float>>>& AIDetectionQueue, 
                       ThreadSafeQueue<cv::Mat>& framesQueue, 
                       ThreadSafeQueue<cv::Mat>& tcpOutputQueue, 
                       int tcpPort, 
                       ThreadSafeQueue<std::string>& commandsQueue,
                       ThreadSafeQueue<std::string>& faultsQueue)
    : cameraComponent(cameraQueue, commandsQueue, faultsQueue),
      faceDetectionComponent(cameraQueue, faceDetectionQueue, faceRectQueue, commandsQueue, faultsQueue),
      AiComponent(faceDetectionQueue, faceRectQueue, AIDetectionQueue, framesQueue, commandsQueue, faultsQueue),
      tcpComponent(tcpPort, framesQueue, AIDetectionQueue, commandsQueue, faultsQueue),
      cameraQueue(cameraQueue), 
      faceDetectionQueue(faceDetectionQueue), 
      faceRectQueue(faceRectQueue),
      AIDetectionQueue(AIDetectionQueue),
      framesQueue(framesQueue), 
      tcpPort(tcpPort), 
      tcpOutputQueue(tcpOutputQueue), 
      commandsQueue(commandsQueue),
      faultsQueue(faultsQueue),
      running(false), 
      firstRun(true) 
      {
        func_ptrs[SET_FPS]      = &DMSManager::setFPS;
        func_ptrs[SET_FDT]      = &DMSManager::setFDT;
        func_ptrs[TURN_SYSTEM]  = &DMSManager::turnSystem;
        func_ptrs[SET_FD_MODEL] = &DMSManager::setFDmodel;
        func_ptrs[SET_HP_MODEL] = &DMSManager::setHPmodel;
        func_ptrs[SET_EG_MODEL] = &DMSManager::setEGmodel;
        func_ptrs[SET_SOURCE]   = &DMSManager::setSource;
      }

// Destructor (cleanup)
DMSManager::~DMSManager() {
    stopSystem();
}

// Startup system
bool DMSManager::startSystem() {
    if (running) return false;  // Prevent the system from starting if it's already running
    running = true;

    if (firstRun) {
        // Starting each component in its own thread
        cameraThread = std::thread(&DMSManager::cameraLoop, this);  // Start the camera loop in its own thread
        faceDetectionThread = std::thread(&DMSManager::faceDetectionLoop, this);  // Start face detection in its own thread
        AIThread = std::thread(&DMSManager::AILoop, this);  // Start AI detection in its own thread
        tcpThread = std::thread(&DMSManager::commtcpLoop, this); // Start tcp thread in its own thread
        commandsThread = std::thread(&DMSManager::commandsLoop, this); // Start commands thread in its own thread
        firstRun = false;
        return true;
    } else {
        // restart threads
        cameraThread = std::thread(&DMSManager::cameraLoop, this);
        faceDetectionThread = std::thread(&DMSManager::faceDetectionLoop, this);
        AIThread = std::thread(&DMSManager::AILoop, this);
        firstRun = false;
        return true;
    }
}

// Stop the system
void DMSManager::stopSystem() {
    running = false;  // Signal all loops to stop
    clearQueues();

    // Log performance metrics
    AiComponent.logPerformanceMetrics();
    tcpComponent.logDataTransferMetrics();
    faceDetectionComponent.logPerformanceMetrics();

    // Stop components
    cameraComponent.stopCapture();
    faceDetectionComponent.stopDetection();
    AiComponent.stopAIDetection();

    // Wait for each thread to finish its task and join
    if (cameraThread.joinable()) cameraThread.join();
    if (faceDetectionThread.joinable()) faceDetectionThread.join();
    if (AIThread.joinable()) AIThread.join();


}

// Setup signal handlers
void DMSManager::setupSignalHandlers() {
    signal(SIGPIPE, SIG_IGN);
}

// Component loops that start in their own thread
void DMSManager::cameraLoop() {
    cameraComponent.startCapture();
}

void DMSManager::faceDetectionLoop() {
    faceDetectionComponent.startDetection();
}

void DMSManager::AILoop() {
    AiComponent.startAIDetection();
}

void DMSManager::commtcpLoop() {
    setupSignalHandlers();
    tcpComponent.startServer();
}

// Loop for DMSManager component to check for any needed commands by components
void DMSManager::commandsLoop(){
    std::string command;
    while (true) {
        if (commandsQueue.tryPop(command)) {
            std::cout << "Received command in the DMS manager: " << command << std::endl;
            this->handleCommand(command);
        }
    }
}

// Functions for different configurations
void DMSManager::setCameraFPS(int fps) {
    cameraComponent.setFPS(fps);
}

void DMSManager::setFaceFDT(int fdt) {
    faceDetectionComponent.setFDT(fdt);
}

void DMSManager::setCamereSource(const std::string& source) {
    cameraComponent.setSource(source);
    clearQueues();
}

void DMSManager::clearQueues(){
    cameraQueue.clear();
    faceDetectionQueue.clear();
    faceRectQueue.clear();
    AIDetectionQueue.clear();
    framesQueue.clear();
    tcpOutputQueue.clear();
}

void DMSManager::handleCommand(std::string& command) {
    clearQueues();
    //model paths
    std::map<HEADPOSE_MODELS, std::string> headPoseModels = {
        {AX, "/home/dms/DMS/ModularCode/include/Ax.engine"},
        {AY, "/home/dms/DMS/ModularCode/include/Ay.engine"},
        {AZ, "/home/dms/DMS/ModularCode/include/Az.engine"},
        {A0, "/home/dms/DMS/ModularCode/include/A0.engine"},
        {eff0, "/home/dms/DMS/ModularCode/include/eff0.engine"},
        {eff1, "/home/dms/DMS/ModularCode/include/eff1.engine"},
        {eff2, "/home/dms/DMS/ModularCode/include/eff2.engine"},
        {eff3, "/home/dms/DMS/ModularCode/include/eff3.engine"},
        {whenNet, "/home/dms/DMS/ModularCode/include/gayarNet.engine"},
        {NoHeadPose, "No Head Pose"}
    };

    std::map<EYEGAZE_MODELS, std::string> eyeGazeModels = {
        {mobilenetv3, "/home/dms/DMS/ModularCode/include/mobileNetNew.engine"},
        {squeezenet, "/home/dms/DMS/ModularCode/modelconfigs/squeezenet.engine"},
        {resnet, "/home/dms/DMS/ModularCode/include/resnet_engine.engine"},
        {mobilenet, "/home/dms/DMS/ModularCode/include/mobilenet_engine.engine"},
        {NoEyeGaze, "No Eye Gaze"}
    };

    std::map<FACEDETECTION_MODELS, std::pair<std::string, std::string>> faceDetectionModels = {
        {YoloV3Tiny, {"/home/dms/DMS/ModularCode/modelconfigs/face-yolov3-tiny.cfg", "/home/dms/DMS/ModularCode/modelconfigs/face-yolov3-tiny_41000.weights"}},
        {YoloV2, {"/home/dms/DMS/ModularCode/modelconfigs/yoloface-500k-v2.cfg", "/home/dms/DMS/ModularCode/modelconfigs/yoloface-500k-v2.weights"}},
        {NoFaceDetection, {"No Face Detection", "No Face Detection"}}
    };

    if (command.starts_with("SET_SOURCE:")) 
    {
        // Handle the SET_SOURCE command as a string
        setSource(command);
    } 
    else if (command == "Clear Queue") {
        // Handle the Clear Queue command as a string
        clearQueue();
    }
    else
    {
        unsigned short cmd = *reinterpret_cast<const unsigned short*>(command.data()); //treats the pointer to the string data as a pointer to an unsigned short
        //reads the first two bytes of the string data as an unsigned short value, effectively converting the string data to a hex command.
        
        unsigned short func_to_call = cmd & 0x000F; //extract the last 4 bits
        // Check if func_to_call is within the bounds of the array
        
        if (func_to_call < COMMANDS_NUM && func_ptrs[func_to_call] != NULL)
        {
            // Call the function and get the output
            Std_ReturnType output = func_ptrs[func_to_call](cmd);
            if(output == E_NOT_OK)
            {
                std::cout << "Something went wrong with the command\n" << std::endl;
            }
            else
            {
                //do nth
            }
        }
        else
        {
            std::cerr << "Unknown command: " << command << std::endl;
        }
    }
}

Std_ReturnType DMSManager::setFPS(command)
{
    unsigned short fpsValue= (command >> 4) & (0x00FF);
    if(fpsValue>=0 && fpsValue<=100)
    {
        std::cout << "Setting FPS to: " << fpsValue << std::endl;
        setCameraFPS(fpsValue);
        return E_OK;
    }
    else
    {
        std::cerr << "Invalid SETFPS command format: " << command << std::endl;
        return E_NOT_OK;
    }
}

Std_ReturnType DMSManager::setFDT(command)
{
    unsigned short fdtValue= (command >> 4) & (0x00FF);
    if(fdtValue>=0 && fdtValue<=100)
    {
        std::cout << "Setting FDT to: " << fdtValue << std::endl;
        setFaceFDT(fdtValue);
        return E_OK;
    }
    else
    {
        std::cerr << "Invalid SETFDT command format: " << command << std::endl;
        return E_NOT_OK;
    }
}


Std_ReturnType DMSManager::turnSystem(command)
{
    unsigned short system = (command >> 4) & (0x00FF);
    if (system == TURN_SYSTEM_ON)
    {
        std::cout << "Turning on..." << std::endl;
        startSystem();
        return E_OK;
    }
    else if (system == TURN_SYSTEM_OFF)
    {
        std::cout << "Turning off..." << std::endl;
        stopSystem();
        return E_OK
    }
    else
    {
        return E_NOT_OK;
    }
    
}

Std_ReturnType DMSManager::setFDmodel(command)
{
    unsigned short modelValue = (command >> 4) & (0x00FF);
    auto it = faceDetectionModels.find(modelValue);
    if (it != faceDetectionModels.end()) 
    {
        std::string weightPath = it->second.second;
        std::string configPath = it->second.first;
        if (weightPath == "No Face Detection" && configPath == "No Face Detection") 
        {
            faceDetectionComponent.modelstatus = false;
            std::cout << "Updated Face Detection Model and Config to: " << weightPath << " and " << configPath << std::endl;
            clearQueues();
        } 
        else 
        {
            faceDetectionComponent.stopDetection();
            faceDetectionComponent.initialize(configPath, weightPath);
            faceDetectionComponent.modelstatus = true;
            faceDetectionComponent.startDetection();
            std::cout << "Updated Face Detection Model and Config to: " << weightPath << " and " << configPath << std::endl;
            clearQueues();
        }
        return E_OK;
    } 
    else 
    {
        std::cerr << "Face detection model identifier not recognized: " << modelValue << std::endl;
        return E_NOT_OK;
    }
}

Std_ReturnType DMSManager::setHPmodel(command)
{
    unsigned short modelValue = (command >> 4) & (0x00FF);
    std::cout << "Setting Head Pose Model to: " << modelValue << std::endl;
    if (headPoseModels.find(modelValue) != headPoseModels.end()) 
    {
        clearQueues();
        AiComponent.updateHeadPoseEngine(headPoseModels[modelValue]);
        clearQueues();
        return E_OK;
    } 
    else 
    {
        std::cerr << "Head pose model identifier not recognized: " << modelValue << std::endl;
        return E_NOT_OK;
    }
}

Std_ReturnType DMSManager::setEGmodel(command)
{
    unsigned short modelValue = (command >> 4) & (0x00FF);
    std::cout << "Setting Eye Gaze Model to: " << modelValue << std::endl;
    if (eyeGazeModels.find(modelValue) != eyeGazeModels.end()) 
    {
        clearQueues();
        AiComponent.updateEyeGazeEngine(eyeGazeModels[modelValue]);
        clearQueues();
        return E_OK;
    } 
    else 
    {
        std::cerr << "Eye gaze model identifier not recognized: " << modelValue << std::endl;
        return E_NOT_OK;
    }
}

Std_ReturnType DMSManager::setSource(command)
{
    // Setting source
    size_t pos = command.find(":");
    if (pos != std::string::npos) 
    {
        std::string sourceStr = command.substr(pos + 1);
        if (sourceStr == "camera") 
        {
            sourceStr = "/dev/video0";  // or the appropriate camera source for your system
        } 
        else if (sourceStr.find("video:") == 0) 
        {
            std::string videoName = sourceStr.substr(6);  // Extract the video name after "video:"
            sourceStr = "/home/dms/DMS/Videos/" + videoName;  // Construct the full path to the video file
        } 
        else 
        {
            std::cerr << "Invalid source value: " << sourceStr << std::endl;
            return E_NOT_OK;
        }
        std::cout << "Setting Source to: " << sourceStr << std::endl;
        setCamereSource(sourceStr);
        return E_OK;
    } 
    else 
    {
        std::cerr << "Invalid SET_SOURCE command format: " << command << std::endl;
        return E_NOT_OK;
    }
    
}

// Initialization functions needed for some components
bool DMSManager::initializeCamera(const std::string& source) {
    return cameraComponent.initialize(source);
}
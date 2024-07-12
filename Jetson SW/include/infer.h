#ifndef INFER_H
#define INFER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <mutex>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <unistd.h>

struct CPUUsage {
    long idleTime;
    long totalTime;
};

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
} gLogger;

class TRTEngineSingleton {
private:
    static TRTEngineSingleton* instance;
    static std::mutex mtx; // Mutex for thread safety

    ICudaEngine* engine1; // Head pose engine
    ICudaEngine* engine2; // Eye gaze engine

    size_t peakHeadPoseGpuMemoryUsage = 0;
    size_t peakEyeGazeGpuMemoryUsage = 0;
    size_t totalCpuMemoryUsageHeadPose = 0;
    size_t totalCpuMemoryUsageEyeGaze = 0;
    size_t headPoseInferenceCount = 0;
    size_t eyeGazeInferenceCount = 0;
    size_t totalCpuUsageEyeGaze = 0;
    size_t totalCpuUsageHeadPose = 0;

    TRTEngineSingleton();

public:
    static TRTEngineSingleton* getInstance();
    void loadEngines();
    void setEngine1(const std::string& enginePath);
    void setEngine2(const std::string& enginePath);
    std::vector<float> inferHeadPose(const cv::Mat& croppedFace);
    std::vector<float> inferEyeGaze(const cv::Mat& croppedFace);
    size_t getPeakHeadPoseGpuMemoryUsage() const;
    size_t getPeakEyeGazeGpuMemoryUsage() const;
    size_t gettotalCpuMemoryUsageHeadPose() const;
    size_t gettotalCpuMemoryUsageEyeGaze() const;
    size_t getheadPoseInferenceCount() const;
    size_t geteyeGazeInferenceCount() const;
    size_t geteyeGazeCpuUsage() const;
    size_t getheadPoseCpuUsage() const;
    void resetPeakGpuMemoryUsage();
    size_t getHostMemoryUsage();
    size_t getPageFaults();
    CPUUsage getCPUUsage();
    double calculateCPUUsage(const CPUUsage& prev, const CPUUsage& curr);
    ~TRTEngineSingleton();

private:
    ICudaEngine* loadEngine(const std::string& engineFile);
};

#endif // INFER_H

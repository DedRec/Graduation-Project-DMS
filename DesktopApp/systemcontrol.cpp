#include "systemcontrol.h"
#include "ui_systemcontrol.h"
#include <QButtonGroup>


SystemControl::SystemControl(DataHandler *dataHandler, ConnectionWidget *connectionWidget, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::SystemControl),
    dataHandler(dataHandler),
    connectionWidget(connectionWidget)
{
    ui->setupUi(this);
    connect(ui->sendtrialbutton, &QPushButton::clicked, this, &SystemControl::onSendButtonClicked);
    // Connect the ON button signal to the slot
    connect(ui->onbutton, &QPushButton::clicked, this, &SystemControl::onSystemOnClicked);

    // Connect the OFF button signal to the slot
    connect(ui->offbutton, &QPushButton::clicked, this, &SystemControl::onSystemOffClicked);

    // Populate the combo boxes with model options
    ui->comboFaceDetection->addItems({"YoloV3 Tiny", "YoloV2", "No Face Detection"});
    ui->comboHeadPose->addItems({"AX", "AY", "AZ", "A0","eff0","eff1","eff2","eff3","No Head Pose"});
    ui->comboEyeGaze->addItems({"mobilenetv3", "squeezenet", "resnet", "mobilenet", "No Eye Gaze"});
    // Set default initial values
    ui->comboFaceDetection->setCurrentIndex(ui->comboFaceDetection->findText("YoloV2"));
    ui->comboHeadPose->setCurrentIndex(ui->comboHeadPose->findText("AY"));
    ui->comboEyeGaze->setCurrentIndex(ui->comboEyeGaze->findText("mobilenetv3"));

}

SystemControl::~SystemControl()
{
    delete ui;
}

void SystemControl::onSendButtonClicked() {
    if (!connectionWidget->isConnected()) {
        QMessageBox::warning(this, "Connection Error", "The system is disconnected. Please connect before proceeding.");
        return;
    }
    if (!systemStatus) {
        QMessageBox::warning(this, "System Status", "The system is currently off. Please turn it on before proceeding.");
        return;
    }
    // Retrieve selected options from the combo boxes
    QString faceDetectionModel = ui->comboFaceDetection->currentText();
    QString headPoseModel = ui->comboHeadPose->currentText();
    QString eyeGazeModel = ui->comboEyeGaze->currentText();

    // Construct hex commands based on selected models
    unsigned char fdCommand = 0x00; // Initialize to zero for face detection
    if (faceDetectionModel == "YoloV3 Tiny") {
        fdCommand = 0x01; // YoloV3 Tiny
    }
    else if (faceDetectionModel == "YoloV2") {
        fdCommand = 0x02; // YoloV2
    }
    else {
        fdCommand = 0x00; // No Face Detection
    }
    fdCommand |= SET_FD_MODEL_BASE; // Set the base value (0x3)

    // Construct head pose command
    unsigned char hpCommand = 0x00; // Initialize to zero for head pose
    if (headPoseModel == "AX") {
        hpCommand = 0x01; // AX
    }
    else if (headPoseModel == "AY") {
        hpCommand = 0x02; // AY
    }
    else if (headPoseModel == "AZ") {
        hpCommand = 0x03; // AZ
    }
    else if (headPoseModel == "A0") {
        hpCommand = 0x04; // A0
    }
    else if (headPoseModel == "eff0") {
        hpCommand = 0x05; // eff0
    }
    else if (headPoseModel == "eff1") {
        hpCommand = 0x06; // eff1
    }
    else if (headPoseModel == "eff2") {
        hpCommand = 0x07; // eff2
    }
    else if (headPoseModel == "eff3") {
        hpCommand = 0x08; // eff3
    }
    else {
        hpCommand = 0x09; // No Head Pose
    }
    hpCommand |= SET_HP_MODEL_BASE; // Set the base value (0x4)

    // Construct eye gaze command
    unsigned char egCommand = 0x00; // Initialize to zero for eye gaze
    if (eyeGazeModel == "mobilenetv3") {
        egCommand = 0x01; // mobilenetv3
    }
    else if (eyeGazeModel == "squeezenet") {
        egCommand = 0x02; // squeezenet
    }
    else if (eyeGazeModel == "resnet") {
        egCommand = 0x03; // resnet
    }
    else if (eyeGazeModel == "mobilenet") {
        egCommand = 0x04; // mobilenet
    }
    else {
        egCommand = 0x05; // No Eye Gaze
    }
    egCommand |= SET_EG_MODEL_BASE; // Set the base value (0x5)

    // Create QByteArrays for hex commands
    QByteArray fdData;
    fdData.append(fdCommand);

    QByteArray hpData;
    hpData.append(hpCommand);

    QByteArray egData;
    egData.append(egCommand);

    // Use DataHandler to send the messages
    dataHandler->sendData(fdData);
    dataHandler->sendData(hpData);
    dataHandler->sendData(egData);

}



void SystemControl::onSystemOnClicked() {
    if (!connectionWidget->isConnected()) {
        QMessageBox::warning(this, "Connection Error", "The system is disconnected. Please connect before proceeding.");
        return;
    }
    unsigned char message = TURN_ON_COMMAND;
    QByteArray data;
    data.append(message);

    dataHandler->sendData(data);
    systemStatus = true;
}

void SystemControl::onSystemOffClicked() {
    if (!connectionWidget->isConnected()) {
        QMessageBox::warning(this, "Connection Error", "The system is disconnected. Please connect before proceeding.");
        return;
    }

    unsigned char message = TURN_OFF_COMMAND;
    QByteArray data;
    data.append(message);

    dataHandler->sendData(data);
    systemStatus = false;
}



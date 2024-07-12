#ifndef SYSTEMCONTROL_H
#define SYSTEMCONTROL_H

#include <QWidget>
#include "datahandler.h"
#include <QMessageBox>
#include "connectionwidget.h"
#include "ui_systemcontrol.h"

#define SET_FD_MODEL_BASE      (0x3)  // Base value for face detection model commands
#define SET_HP_MODEL_BASE      (0x4)  // Base value for head pose model commands
#define SET_EG_MODEL_BASE      (0x5)  // Base value for eye gaze model commands
#define TURN_ON_COMMAND        (0x12)
#define TURN_OFF_COMMAND       (0x22)

namespace Ui {
class SystemControl;
}

class SystemControl : public QWidget
{
    Q_OBJECT

public:
    explicit SystemControl(DataHandler *dataHandler,ConnectionWidget *connectionWidget, QWidget *parent = nullptr);
    ~SystemControl();
    bool systemStatus=true;
    QString getActiveFaceModel() const { return ui->comboFaceDetection->currentText(); }
    QString getActiveHeadModel() const { return ui->comboHeadPose->currentText(); }
    QString getActiveEyeModel() const { return ui->comboEyeGaze->currentText(); }


private slots:
    void onSystemOnClicked();
    void onSystemOffClicked();
    void onSendButtonClicked();

private:
    Ui::SystemControl *ui;
    DataHandler *dataHandler;
    ConnectionWidget *connectionWidget;

};

#endif // SYSTEMCONTROL_H

#include "configswidget.h"
#include "ui_configswidget.h"

ConfigsWidget::ConfigsWidget(DataHandler *dataHandler, ConnectionWidget *connectionWidget, SystemControl *systemControl, QWidget *parent)
    : QWidget(parent),
    ui(new Ui::ConfigsWidget),
    dataHandler(dataHandler),
    connectionWidget(connectionWidget),
    systemControl(systemControl)
{
    ui->setupUi(this);

    // Add initial values to the combo box
    ui->source_combo->addItem("Camera");
    ui->source_combo->addItem("Video");

    // Hide the line edit by default
    ui->videoname->setVisible(false);
    ui->label_video->setVisible(false);

    // Connect the combo box signal to the slot
    connect(ui->source_combo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &ConfigsWidget::onCaptureSourceChanged);

    // Connect the apply changes button to the slot
    connect(ui->applybutton, &QPushButton::clicked, this, &ConfigsWidget::onApplyChangesClicked);
}

ConfigsWidget::~ConfigsWidget()
{
    delete ui;
}

void ConfigsWidget::onCaptureSourceChanged(int index) {
    if (ui->source_combo->currentText() == "Video") {
        ui->videoname->setVisible(true);
        ui->label_video->setVisible(true);
    } else {
        ui->videoname->setVisible(false);
        ui->label_video->setVisible(false);
    }
}

void ConfigsWidget::onApplyChangesClicked() {
    if (!connectionWidget->isConnected()) {
        QMessageBox::warning(this, "Connection Error", "The system is disconnected. Please connect before proceeding.");
        return;
    }
    if (!systemControl->systemStatus) {
        QMessageBox::warning(this, "System Status", "The system is currently off. Please turn it on before proceeding.");
        return;
    }

    // Retrieve values from UI components
    int fps = ui->fps_spin->value();
    int fdt = ui->fdt_spin->value();
    QString source;

    if (ui->source_combo->currentText() == "Camera") {
        source = "camera";
    } else {
        source = "video:" + ui->videoname->text();
    }

    // Define command codes
    const int SET_FPS = 0x0000;
    const int SET_FDT = 0x0001;

    // Create byte arrays to hold the messages
    QByteArray fpsMessage;
    QByteArray fdtMessage;
    QString sourceMessage = "SET_SOURCE:" + source;
    QByteArray sourceData = sourceMessage.toUtf8();

    // Create combined FPS command and value
    int fpsValue = (fps & 0xFF) << 4 | SET_FPS; // Combine value and command
    fpsMessage.append(reinterpret_cast<const char*>(&fpsValue), sizeof(fpsValue));

    // Create combined FDT command and value
    int fdtValue = (fdt & 0xFF) << 4 | SET_FDT; // Combine value and command
    fdtMessage.append(reinterpret_cast<const char*>(&fdtValue), sizeof(fdtValue));

    // Prefix source message with its length
    int sourceSize = sourceData.size();
    sourceData.prepend(reinterpret_cast<const char*>(&sourceSize), sizeof(int));

    // Use DataHandler to send the messages
    dataHandler->sendData(fpsMessage);
    dataHandler->sendData(fdtMessage);
    dataHandler->sendData(sourceData);
}


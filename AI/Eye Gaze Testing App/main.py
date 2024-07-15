from ast import Break
import sys
import cv2
import time
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image
from torchvision import transforms
from models import MobileNetV3Small, ResNet50, MobileNetV2, SqueezeNet
from ultralytics import YOLO
import torch

class Ui_EyeGazeTesting(object):
    def setupUi(self, EyeGazeTesting):
        EyeGazeTesting.setObjectName("EyeGazeTesting")
        EyeGazeTesting.resize(1002, 596)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(228, 229, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(241, 242, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(114, 114, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(152, 153, 170))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(228, 229, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(241, 242, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(228, 229, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(241, 242, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(114, 114, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(152, 153, 170))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(228, 229, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(241, 242, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(114, 114, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(228, 229, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(241, 242, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(114, 114, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(152, 153, 170))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(114, 114, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(114, 114, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(228, 229, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(228, 229, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(228, 229, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipText, brush)
        EyeGazeTesting.setPalette(palette)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icons/eyegaze.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        EyeGazeTesting.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(EyeGazeTesting)
        self.centralwidget.setObjectName("centralwidget")

        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 711, 541))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")

        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.webCamLabel = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.webCamLabel.setObjectName("webCamLabel")
        self.horizontalLayout.addWidget(self.webCamLabel)

        # Get the size of the horizontalLayoutWidget
        layout_size = self.horizontalLayoutWidget.size()

        # Create a black QPixmap with the same size as the horizontalLayoutWidget
        black_pixmap = QPixmap(layout_size)
        black_pixmap.fill(QColor('black'))
        # Set the black QPixmap to the label
        self.webCamLabel.setPixmap(black_pixmap)

        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(739, 9, 251, 541))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_3 = QtWidgets.QLabel(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)

        self.Models = QtWidgets.QComboBox(self.verticalLayoutWidget)
        self.Models.setObjectName("Models")
        self.verticalLayout.addWidget(self.Models)
        self.Models.addItems(["MobileNetV3_small", "MobileNetV2", "SqueezeNet","ResNet50"])

        self.label_2 = QtWidgets.QLabel(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.pushButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.pushButton.setFont(font)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("icons/videoTest.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton.setIcon(icon1)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout.addWidget(self.pushButton)
        self.Picture = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.Picture.setFont(font)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("icons/pictureTest.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Picture.setIcon(icon2)
        self.Picture.setObjectName("Picture")
        self.verticalLayout.addWidget(self.Picture)
        self.Folder = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.Folder.setFont(font)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("icons/Folder.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Folder.setIcon(icon3)
        self.Folder.setObjectName("Folder")
        self.verticalLayout.addWidget(self.Folder)
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)

        self.WebCam = QtWidgets.QPushButton(self.verticalLayoutWidget) #webcam button
        self.WebCam.clicked.connect(self.ModelSimu)

        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.WebCam.setFont(font)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("icons/webcamTest.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.WebCam.setIcon(icon4)
        self.WebCam.setObjectName("WebCam")
        self.verticalLayout.addWidget(self.WebCam)
        self.label_4 = QtWidgets.QLabel(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.verticalLayout.addWidget(self.label_4)
        self.spinBox = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.spinBox.setObjectName("spinBox")
        self.verticalLayout.addWidget(self.spinBox)

        self.Stop = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.Stop.clicked.connect(self.stop_webcam)
        
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.Stop.setFont(font)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("icons/stopVideo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Stop.setIcon(icon5)
        self.Stop.setObjectName("Stop")
        self.verticalLayout.addWidget(self.Stop)
        EyeGazeTesting.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(EyeGazeTesting)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1002, 21))
        self.menubar.setObjectName("menubar")
        EyeGazeTesting.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(EyeGazeTesting)
        self.statusbar.setObjectName("statusbar")
        EyeGazeTesting.setStatusBar(self.statusbar)

        self.retranslateUi(EyeGazeTesting)
        QtCore.QMetaObject.connectSlotsByName(EyeGazeTesting)

    def retranslateUi(self, EyeGazeTesting):
        _translate = QtCore.QCoreApplication.translate
        EyeGazeTesting.setWindowTitle(_translate("EyeGazeTesting", "MainWindow"))
        self.label_3.setText(_translate("EyeGazeTesting", "Choose Model"))
        self.label_2.setText(_translate("EyeGazeTesting", "Test On"))
        self.pushButton.setText(_translate("EyeGazeTesting", "Video"))
        self.Picture.setText(_translate("EyeGazeTesting", "Picture"))
        self.Folder.setText(_translate("EyeGazeTesting", "Testing Folder"))
        self.label.setText(_translate("EyeGazeTesting", "Test In Real Time"))
        self.WebCam.setText(_translate("EyeGazeTesting", "WebCam"))
        self.label_4.setText(_translate("EyeGazeTesting", "Yolo Croppping Ratio"))
        self.Stop.setText(_translate("EyeGazeTesting", "Stop"))
    
    def startWebCam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return

        self.timer.start(20)  

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to RGB format
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            qt_image = qt_image.scaled(640, 480, Qt.KeepAspectRatio) 
            self.webCamLabel.setPixmap(QPixmap.fromImage(qt_image))

    def stop_webcam(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()


    def ModelSimu(self):
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return

        # YOLO model 
        model_yolo = YOLO("weights/yolov8n-face.pt")

        model_mnv3 = MobileNetV3Small()
        model_mnv2 = MobileNetV2()
        model_sqz = SqueezeNet()
        model_resnet = ResNet50()

        model_mnv3.load_weights("weights/MNV3_small_weights_cleaned_100.pth")
        #model_mnv2.load_weights("MNV3_small_weights_cleaned_100.pth")
        model_sqz.load_weights("weights/Squeezenet.pt")
        #model_resnet.load_weights("MNV3_small_weights_cleaned_100.pth")

        chosen_model = None
        if self.Models.currentText() == "MobileNetV3_small":
            chosen_model = model_mnv3
        elif self.Models.currentText() == "MobileNetV2":
            chosen_model = model_mnv2
        elif self.Models.currentText() == "SqueezeNet":
            chosen_model = model_sqz
        elif self.Models.currentText() == "ResNet50":
            chosen_model = model_sqz

        # Eye Gaze Detection model
        # model_mnv3 = MobileNetV3Small()
        # model_mnv3.load_weights("MNV3_small_weights_cleaned_100.pth")

        eyeGazeClasses = ['Eyes Closed', 'Forward', 'Left Mirror', 'Radio', 'Rearview', 'Right Mirror', 'Shoulder', 'Speedometer']

        start_time = time.time()
        frame_count = 0

        while self.cap is not None: # while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break

            results = model_yolo(frame)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    height = y2 - y1
                    cropped_height = int(self.spinBox.value() * 0.01 * height) # best ranges from 50 - 55

                    y1_cropped = y1
                    y2_cropped = y1 + cropped_height

                    # Draw bounding box on the frame
                    cv2.rectangle(frame, (x1, y1_cropped), (x2, y2_cropped), (255, 0, 255), 3)

                    # Extract the cropped face region from the frame
                    face_cropped = frame[y1_cropped:y2_cropped, x1:x2]

                    if face_cropped.size != 0:
                        pil_image = Image.fromarray(cv2.cvtColor(face_cropped, cv2.COLOR_BGR2RGB))
                        input_image = transform(pil_image)
                        input_tensor = input_image.unsqueeze(0)

                        start_time = time.time()
                        with torch.no_grad():
                            output = chosen_model(input_tensor)
                        inference_time = time.time() - start_time

                        _, predicted_idx = torch.max(output, 1)
                        predicted_label = eyeGazeClasses[predicted_idx]

                        # Display the predicted label on the frame
                        cv2.putText(frame, predicted_label, (x1, y1_cropped - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,215, 255), 2)

                        frame_count += 1
                        # if frame_count % 10 == 0:  # Calculate FPS every 10 frames
                        #     elapsed_time = time.time() - start_time
                        # Display inference time
                        cv2.putText(frame, f"Model: {self.Models.currentText()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (187,104,199), 2)
                        cv2.putText(frame, f"Inference Time: {inference_time:.2f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (187,104,199), 2)

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            qt_image = qt_image.scaled(640, 480, Qt.KeepAspectRatio)
            self.webCamLabel.setPixmap(QPixmap.fromImage(qt_image))

            if cv2.waitKey(20) & 0xFF == ord('q'):
                Break

        # self.cap.release()
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    EyeGazeTesting = QtWidgets.QMainWindow()
    ui = Ui_EyeGazeTesting()
    ui.setupUi(EyeGazeTesting)
    EyeGazeTesting.show()
    sys.exit(app.exec_())
from PyQt5.QtGui import QPixmap
import numpy as np
from window import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QInputDialog
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2
from hand import HandDetector
from predict import predict
import time


def label_show(label, image):
    qt_img_buf = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    qt_img = QtGui.QImage(qt_img_buf.data, qt_img_buf.shape[1], qt_img_buf.shape[0], QtGui.QImage.Format_RGB32)
    image = QPixmap.fromImage(qt_img).scaled(label.width(), label.height())
    label.setPixmap(image)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.image_file = False
        self.setupUi(self)
        self.initUI()
        self.hand_size = 0.6
        self.detector = HandDetector()
        self.videoFPS = 24
        self.image = False
        self.video = False
        self.camera = False
        self.camera_selected = 0
        self.detect = False
        self.model = predict(model='models/inference_model1')
        self.cap = None
        self.predict = False

    def maya(self, word=None, speed=200, action=True):
        gif = QtGui.QMovie(f'src/{word}.gif')
        self.labelMAYA.setMovie(gif)
        gif.setSpeed(speed)
        if action:
            gif.start()
        else:
            gif.stop()

    def print_log(self, log_words):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.textLog.append(current_time + ':' + log_words)  # 在指定的区域显示提示信息
        cursor = self.textLog.textCursor()
        self.textLog.moveCursor(cursor.End)  # 光标移到最后，这样就会自动显示出来
        QtWidgets.QApplication.processEvents()

    def initUI(self):
        self.setWindowTitle('手语识别可视化界面')

        self.actionLoadImage.triggered.connect(self.load_image)
        self.actionStart.triggered.connect(self.start_detect)
        self.actionLoadVideo.triggered.connect(self.load_video)
        self.actionVideoFPS.triggered.connect(self.change_VideoFPS)
        self.actionOpenCamera.triggered.connect(self.open_camera)
        self.actionCloseCamera.triggered.connect(self.close_camera)
        self.actionPredict.triggered.connect(self.predict)
        self.actionSelectedCamera.triggered.connect(self.select_camera)
        self.actionHandSize.triggered.connect(self.changeHandSize)
        self.actionModelFile.triggered.connect(self.changeModelFile)
        self.actionMinDetectionConfidence.triggered.connect(self.changeMinDetectionConfidence)
        self.actionMinTrackingConfidence.triggered.connect(self.changeMinTrackingConfidence)
        self.actionExit.triggered.connect(self.exit)

    def exit(self):
        app = QApplication.instance()
        # 退出应用程序
        app.quit()

    def load_image(self):
        image_name, image_type = QFileDialog.getOpenFileName(self, '打开图片', r'./', '图片 (*.png *.jpg *.jpeg)')
        cv2_image = cv2.imread(image_name)
        label_show(label=self.labelImageOrVideo, image=cv2_image)
        self.image = True
        self.print_log(log_words=f'load image:{image_name}')
        self.image_file = cv2_image

    def load_video(self):
        self.video_name, video_type = QFileDialog.getOpenFileName(self, '打开视频', r'./', '视频 (*.avi *.mp4)')
        self.cap = cv2.VideoCapture(self.video_name)
        self.video = True
        self.print_log(log_words=f'load video:{self.video_name}')
        while True:
            ret, cv2_image = self.cap.read()
            if not ret:
                break
            label_show(label=self.labelImageOrVideo, image=cv2_image)
            cv2.waitKey(int(1000 / self.videoFPS))

    def start_detect(self):
        maya = cv2.imread('src/MAYA_default.png')
        self.detect = True
        if self.image:
            hand, rectangle, success = self.detector.rectangle(self.image_file, hand_size=self.hand_size)
            if success:
                label_show(label=self.labelImageOrVideo, image=rectangle)
            else:
                self.print_log(log_words=f'cant find hand')
        if self.video:
            cap = cv2.VideoCapture(self.video_name)
            while True:
                ret, image = cap.read()
                if not ret:
                    break
                hand, rectangle, success = self.detector.rectangle(image, hand_size=self.hand_size)
                label_show(label=self.labelImageOrVideo, image=rectangle)
                if success:
                    label_show(label=self.labelMAYA, image=hand)
                else:
                    label_show(label=self.labelMAYA, image=maya)
                cv2.waitKey(int(1000 / self.videoFPS))
        if self.camera:
            self.print_log(log_words=f'using camera detecting')

            while True:
                ret, image = self.cap.read()
                if not ret:
                    break
                hand, rectangle, success = self.detector.rectangle(image, hand_size=self.hand_size)
                label_show(label=self.labelImageOrVideo, image=rectangle)
                if success:
                    label_show(label=self.labelMAYA, image=hand)
                else:
                    label_show(label=self.labelMAYA, image=maya)
                cv2.waitKey(int(1000 / self.videoFPS))

    def predict(self):
        self.predict = True
        if self.image:
            hand, rectangle, success = self.detector.rectangle(self.image_file, hand_size=self.hand_size)
            if success:
                word = self.model.result(hand)['category']
                score = self.model.result(hand)['score']
                result_image = QPixmap(f'src/{word}.png')
                label_show(label=self.labelImageOrVideo, image=rectangle)
                self.maya(word=word)
                self.labelResult.setPixmap(result_image)
                self.print_log(log_words='result:' + word + ' score: ' + str(score))
                self.maya(word=word, speed=100)
            else:
                label_show(label=self.labelImageOrVideo, image=rectangle)
                self.print_log(log_words='cant find hand')

        if self.camera:
            flag = 0
            dic_count = {'bu': 0, 'hao': 0, 'yi': 0, 'si': 0}
            while True:
                ret, cv2_image = self.cap.read()
                if not ret:
                    break
                hand, rectangle, success = self.detector.rectangle(cv2_image, hand_size=self.hand_size)
                label_show(label=self.labelImageOrVideo, image=rectangle)

                if success:

                    hand = cv2.resize(hand, (128, 128))
                    word = self.model.result(hand)['category']
                    score = self.model.result(hand)['score']
                    self.print_log(log_words='result:' + word + ' score: ' + str(score))
                    image = QPixmap(f'src/{word}.png')
                    self.labelResult.setPixmap(image)
                    dic_count[word] += 1
                    if sum(dic_count.values()) < 8:
                        word_show = max(dic_count, key=dic_count.get)
                        if flag == 0:
                            self.maya(word=word_show)
                            flag = 1
                    else:
                        dic_count = {'bu': 0, 'hao': 0, 'yi': 0, 'si': 0}
                        flag = 0
                else:
                    label_show(label=self.labelMAYA, image=cv2.imread('src/MAYA_default.png'))
                    label_show(label=self.labelResult, image=cv2.imread('src/result_default.png'))
                cv2.waitKey(int(1000 / self.videoFPS))

    def open_camera(self):
        camera = cv2.VideoCapture(self.camera_selected)
        self.cap = camera
        self.print_log(log_words=f'opening camera')
        while True:
            ret, cv2_image = camera.read()
            if not ret:
                break
            self.camera = True
            label_show(label=self.labelImageOrVideo, image=cv2_image)
            cv2.waitKey(0)
            cv2.waitKey(int(1000 / self.videoFPS))

    def change_VideoFPS(self):
        number, ok = QInputDialog.getInt(self, "input video fps", "(10-60)")
        self.videoFPS = number
        self.print_log(log_words=f'set video fps: {str(number)}')

    def close_camera(self):
        self.camera = False
        self.cap.release()
        self.print_log(log_words='camera has been closed')
        self.maya(action=False)
        label_show(label=self.labelResult, image=cv2.imread('src/result_default.png'))
        label_show(label=self.labelMAYA, image=cv2.imread('src/MAYA_default.png'))
        label_show(label=self.labelImageOrVideo, image=cv2.imread('src/default.png'))

    def select_camera(self):
        if self.camera_selected == 0:
            self.camera_selected = 1
            self.actionSelectedCamera.setText('SelectedCamera 1')
            self.print_log(log_words='camera 1 has been selected')
        else:
            self.camera_selected = 0
            self.actionSelectedCamera.setText('SelectedCamera 0')
            self.print_log(log_words='camera 0 has been selected')

    def changeHandSize(self):
        number, ok = QInputDialog.getDouble(self, "input hand_size", "(0-1)")
        self.hand_size = number
        self.print_log(log_words=f'set hand size: {str(number)}')

    def changeModelFile(self):
        path = QFileDialog.getExistingDirectory(self, "choose model filepath", '/')
        self.model = predict(model=path)
        self.print_log(log_words=f'set model: {path}')

    def changeMinDetectionConfidence(self):
        number, ok = QInputDialog.getDouble(self, "MinDetectionConfidence", "(0-1)")
        self.detector.min_detection_confidence = number
        self.print_log(log_words=f'MinDetectionConfidence: {str(number)}')

    def changeMinTrackingConfidence(self):
        number, ok = QInputDialog.getDouble(self, "MinTrackingConfidence", "(0-1)")
        self.detector.min_tracking_confidence = number
        self.print_log(log_words=f'MinTrackingConfidence: {str(number)}')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = MainWindow()
    MainWindow.show()
    sys.exit(app.exec_())

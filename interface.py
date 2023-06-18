from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import keras
import keras.utils as image
from PyQt5.QtCore import Qt
import numpy as np

model = keras.models.load_model('binary.h5')

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setMaximumSize(QtCore.QSize(800, 600))
        MainWindow.setStyleSheet("background-color: rgb(239, 254, 255);\n"
                                 "")

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(-20, 0, 820, 60))
        self.label.setStyleSheet("background-color: rgb(94, 94, 94);")
        self.label.setText("")
        self.label.setObjectName("label")

        self.photo_label = QtWidgets.QLabel(self.centralwidget)
        self.photo_label.setGeometry(QtCore.QRect(120, 85, 570, 420))
        self.photo_label.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.photo_label.setObjectName("photo_label")

        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(0, 495, 780, 30))
        self.label_3.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_3.setObjectName("label_3")
        self.label_3.setAlignment(Qt.AlignCenter)

        font = QtGui.QFont()
        font.setFamily("Franklin Gothic Heavy")
        font.setPointSize(15)
        self.label_3.setFont(font)

        self.label_persnt = QtWidgets.QLabel(self.centralwidget)
        self.label_persnt.setGeometry(QtCore.QRect(720, 495, 60, 30))
        self.label_persnt.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_persnt.setObjectName("label_persnt")
        self.label_persnt.setAlignment(Qt.AlignCenter)

        fontper = QtGui.QFont()
        fontper.setFamily("Franklin Gothic Heavy")
        fontper.setPointSize(13)
        self.label_persnt.setFont(fontper)

        self.ZagrFoto = QtWidgets.QPushButton(self.centralwidget)
        self.ZagrFoto.setGeometry(QtCore.QRect(0, 540, 800, 60))
        self.ZagrFoto.setMaximumSize(QtCore.QSize(16777215, 16777215))

        self.ZagrFoto.setFont(font)
        self.ZagrFoto.setStyleSheet(
            "background-color: rgb(19, 22, 43);\n"
            "background-color: rgb(94, 94, 94);\n"
            "color: rgb(255, 255, 255);\n"
            "")
        self.ZagrFoto.setObjectName("ZagrFoto")
        self.ZagrFoto.clicked.connect(self.open_file_dialog)

        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Оценка участка"))
        self.label_3.setText(_translate("MainWindow", ""))
        self.label_persnt.setText(_translate("MainWindow", ""))
        self.ZagrFoto.setText(_translate("MainWindow", "Загрузка и Анализ изображения"))

    def open_file_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(None, "Выберите изображение", "",
                                                   "Images (*.png *.jpg *.bmp);;All Files (*)",
                                                   options=options)

        img = image.load_img(file_name, target_size=(750, 500))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        if file_name:
            self.load_image(file_name)

        predictions = model.predict(x)
        prediction_description = ' '.join([str(round(p * 100)) + '%' for p in predictions[0]])
        print(predictions[0][0])
        self.label_persnt.setText(prediction_description)

        if predictions[0][0] > 0.4:
            self.label_3.setText('Участок повышенной пожароопасности')
        else:
            self.label_3.setText('Безопасный участок')

    def load_image(self, file_path):
        pixmap = QtGui.QPixmap(file_path)
        pixmap = pixmap.scaled(self.photo_label.width(), self.photo_label.height(),
                               QtCore.Qt.KeepAspectRatio)
        self.photo_label.setPixmap(pixmap)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())



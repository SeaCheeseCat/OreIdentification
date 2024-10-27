# -*- coding: utf-8 -*-
import sys 
from PIL import Image
import torch.nn as nn
from torchvision import transforms,models
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton,QLabel,QTextEdit,QFileDialog,QHBoxLayout,QVBoxLayout,QSplitter,QComboBox,QSpinBox
from PyQt5.Qt import QWidget, QColor,QPixmap,QIcon,QSize,QCheckBox
from PyQt5 import QtCore, QtGui
import numpy as np
import torch
import os
import torch
import torch.nn as nn
import argparse
from PIL import Image
from torchvision.models import mobilenet_v2


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
classes = ['镜铁矿', '菱镁矿', '黄铁矿', '赤铁矿', '铝土矿', '硬锰矿', '蓝铜矿', '辰砂', '黄铜矿', '钛铁矿', '方铅矿', '斑铜矿', '闪锌矿', '磁铁矿', '锰矿']
#------------------------------------------------------1.加载模型--------------------------------------------------------------
num_classes = 94
class MobileNet(nn.Module):
    def __init__(self, num_classes=num_classes):   
        super(MobileNet, self).__init__()
        net = models.mobilenet_v2(pretrained=True)   
        net.classifier = nn.Sequential()
        self.features = net
        self.classifier = nn.Sequential(   
                nn.Linear(1280, 1000),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(1000, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

path_model="./logs/model4.ckpt"

model=torch.load(path_model)
model = model.to(device)

def get_imageNdarray(imageFilePath):
    input_image = Image.open(imageFilePath).convert("RGB")
    return input_image

def process_imageNdarray(input_image):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_chw = preprocess(input_image)
    return img_chw  






class FirstUi(QMainWindow): 
    def __init__(self):
        super(FirstUi, self).__init__()
        self.init_ui()

    def init_ui(self):
        self.setFixedSize(self.width(), self.height())
        self.setWindowIcon(QtGui.QIcon('./icon.jpg'))
        self.resize(640, 500)  # 设置窗口大小
        self.btn2 = QPushButton('图片识别', self)
        self.btn2.setGeometry(245, 200,150,50)
        self.btn2.clicked.connect(self.slot_btn2_function)
        self.btn_exit = QPushButton('退出', self)
        self.btn_exit.setGeometry(245, 300, 150, 50)
        self.btn_exit.clicked.connect(self.Quit)
        self.label_name = QLabel('welcome 图片识别', self)
        self.label_name.setGeometry(460, 410, 200, 30)

    def Quit(self):
        self.close()

    def slot_btn_function(self):
        self.hide()  
        self.s = write_num()  
        self.s.show()

    def slot_btn2_function(self):
        self.hide() 
        self.s = picture_num()
        self.s.show()

    def __fillColorList(self, comboBox):

        index_black = 0
        index = 0
        for color in self.__colorList:
            if color == "black":
                index_black = index
            index += 1
            pix = QPixmap(70, 20)
            pix.fill(QColor(color))
            comboBox.addItem(QIcon(pix), None)
            comboBox.setIconSize(QSize(70, 20))
            comboBox.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        comboBox.setCurrentIndex(index_black)

    def on_PenColorChange(self):
        color_index = self.__comboBox_penColor.currentIndex()
        color_str = self.__colorList[color_index]

    def on_PenThicknessChange(self):
        penThickness = self.__spinBox_penThickness.value()

    def on_btn_Save_Clicked(self):
        savePath = QFileDialog.getSaveFileName(self, 'Save Your Paint', '.\\', '*.png')
        print(savePath)
        if savePath[0] == "":
            print("Save cancel")
            return

    def Quit(self):
        self.close()



class picture_num(QWidget):
    def __init__(self):
        super(picture_num, self).__init__()
        self.init_ui()

    def init_ui(self):
        self.fname = ''
        self.setWindowIcon(QtGui.QIcon('./icon.jpg'))
        self.resize(520,540)
        self.setFixedSize(self.width(), self.height())
        self.setWindowTitle('图片识别')

        self.label_name5 = QLabel('待载入图片', self)
        self.label_name5.setGeometry(10, 20, 500, 380)
        self.label_name5.setStyleSheet("QLabel{background:gray;}"
                                 "QLabel{color:rgb(0,0,0,120);font-size:15px;font-weight:bold;font-family:宋体;}"
                                 )
        self.label_name5.setAlignment(QtCore.Qt.AlignCenter)
        
        self.edit = QTextEdit(self)
        self.edit.setGeometry(340, 420, 170, 80)

        self.btn_select = QPushButton('选择图片',self)
        self.btn_select.setGeometry(20, 420, 100, 30)
        self.btn_select.clicked.connect(self.select_image)

        self.btn_dis = QPushButton('识别图片',self)
        self.btn_dis.setGeometry(160, 420, 100, 30)
        self.btn_dis.clicked.connect(self.on_btn_Recognize_Clicked)

        self.btn = QPushButton('返回',self)
        self.btn.setGeometry(20, 470, 100, 30)
        self.btn.clicked.connect(self.slot_btn_function)

        self.btn_exit = QPushButton('退出',self)
        self.btn_exit.setGeometry(160, 470, 100, 30)
        self.btn_exit.clicked.connect(self.Quit)

    def select_image(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.png;;*.jpg;;All Files(*)")
        jpg = QtGui.QPixmap(imgName).scaled(self.label_name5.width(), self.label_name5.height())
        self.label_name5.setPixmap(jpg)
        self.fname = imgName

    def on_btn_Recognize_Clicked(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        savePath = self.fname
        input_image = get_imageNdarray(savePath)    
        input_image = input_image.resize((224,224))
        img_chw = process_imageNdarray(input_image)

        if torch.cuda.is_available():
            img_chw = img_chw.view(1, 3, 224, 224).to(device)
        else:
            img_chw = img_chw.view(1, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            torch.no_grad()
            out = model(img_chw)

            score = torch.nn.functional.softmax(out, dim=1)[0] * 100  

            predicted = torch.max(out, 1)[1]

            score = score[predicted.item()].item()    

            txt = str(classes[predicted.item()])


        self.edit.setText('识别结果为:' + str(txt))
        
    def Quit(self):
        self.close()

    def slot_btn_function(self):
        self.hide()
        self.f = FirstUi()
        self.f.show()


def main():
    app = QApplication(sys.argv)
    w = FirstUi()  
    w.show() 
    sys.exit(app.exec_())  


if __name__ == '__main__': 
    main()
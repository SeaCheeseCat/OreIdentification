import cv2
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import QPixmap, QImage, QPalette, QBrush
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt, QUrl
import sys
from torchvision import transforms,models
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton,QLabel,QTextEdit,QFileDialog,QHBoxLayout,QVBoxLayout,QSplitter,QComboBox,QSpinBox
from PyQt5.Qt import QWidget, QColor,QPixmap,QIcon,QSize,QCheckBox
import os
import torch
import torch.nn as nn
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#classes = ['镜铁矿', '菱镁矿', '黄铁矿', '赤铁矿', '铝土矿', '硬锰矿', '蓝铜矿', '辰砂', '黄铜矿', '钛铁矿', '方铅矿', '斑铜矿', '闪锌矿', '磁铁矿', '锰矿']
classes = ['镜铁矿', '钛铁矿', '方铅矿', '斑铜矿', '闪锌矿', '磁铁矿', '锰矿', '菱镁矿', '黄铁矿', '赤铁矿', '铝土矿', '硬锰矿', '蓝铜矿', '辰砂', '黄铜矿']

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

# 获取资源路径
def resource_path(relative_path):
    # 检查是否在打包模式下运行，获取 _MEIPASS 临时路径
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

path_model = resource_path("./logs/model6.ckpt")
#path_model="./logs/model6.ckpt"

#model=torch.load(path_model)
model=torch.load(path_model, map_location='cpu')
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


class StoneRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.capture = cv2.VideoCapture(0)  # 打开摄像头
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)  # 每20ms更新一次摄像头画面
        self.current_frame = None
        self.is_recognizing = False  # 添加一个标志来跟踪识别状态
        # 初始化视频播放器
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.mediaStatusChanged.connect(self.handle_media_status)

    def initUI(self):
        self.setWindowTitle('矿石识别')
        self.setGeometry(100, 100, 800, 600)

        # 设置背景颜色
        self.setStyleSheet("background-color: #F0F0F0;")

        # 设置布局
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        # 设置窗口图标
        self.setWindowIcon(QIcon('res/icon.png'))  # 请替换为图标的实际路径
        #self.set_background_image('res/background.png')
        # 用于显示图像的标签
        self.image_label = QLabel(self)
        self.image_label.setStyleSheet("border: 2px solid #008CBA; border-radius: 10px;")
        self.image_label.setAlignment(Qt.AlignCenter)  # 设置标签内的内容居中对齐
        layout.addWidget(self.image_label)
        self.image_label.setVisible(True)

        # 添加视频播放器
        self.video_widget = QVideoWidget(self)
        self.video_widget.setMinimumSize(640, 480)  # 设置一个最小大小
        layout.addWidget(self.video_widget)
        self.video_widget.setVisible(False)

        # 识别按钮
        self.button = QPushButton('开始识别', self)
        self.button.setFixedHeight(50)
        self.button.setStyleSheet("""
            QPushButton {
                background-color: #008CBA;
                color: white;
                font-size: 16px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #005f7f;
            }
        """)
        self.button.clicked.connect(self.on_recognition_button_clicked)
        layout.addWidget(self.button)

        # 用于显示识别结果的文本框
        self.result_text = QTextEdit(self)
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("""
            QTextEdit {
                font-size: 14px;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: white;
            }
        """)
        layout.addWidget(self.result_text)

        # 添加提示信息的标签
        self.instruction_label = QLabel('将矿石放入白色背景中，同时在摄像头可拍摄范围中间位置', self)
        self.instruction_label.setAlignment(Qt.AlignCenter)  # 设置文本居中
        self.instruction_label.setStyleSheet("""
                   QLabel {
                       font-size: 12px;
                       color: #555555;
                       margin-top: 10px;
                       padding: 5px;
                   }
               """)
        layout.addWidget(self.instruction_label)

        self.setLayout(layout)

        # 定义 result_text 和视频序号的映射关系
        self.video_mapping = {
            '镜铁矿': '1',
            '菱镁矿': '2',
            '黄铁矿': '3',
            '赤铁矿': '4',
            '铝土矿': '5',
            '硬锰矿': '6',
            '蓝铜矿': '7',
            '辰砂': '8',
            '黄铜矿': '9',
            '钛铁矿': '10',
            '方铅矿': '11',
            '斑铜矿': '12',
            '闪锌矿': '13',
            '磁铁矿': '14',
            '锰矿': '15',
        }

    def play_video(self, stone_name):
        # 根据识别结果选择对应的视频
        self.image_label.setVisible(False)
        self.video_widget.setVisible(True)
        video_filetemp = 'res/videos/'+ stone_name +'.mp4'  # 替换为你的实际视频路径
        video_file = resource_path(video_filetemp)
        if not os.path.exists(video_file):
            print("视频文件不存在:", video_file)
            return
        print("播放视频", video_file)
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(video_file)))
        self.media_player.play()  # 播放视频

        #self.media_player.errorOccurred.connect(self.handle_error)

    def set_background_image(self, image_path):
        palette = QPalette()
        palette.setBrush(QPalette.Window, QBrush(QPixmap(image_path)))
        self.setPalette(palette)

    def handle_error(self, error):
        print("播放错误:", error)

    def display_loading(self):
        self.result_text.setPlainText("识别中...")

    def update_frame(self):
        ret, frame = self.capture.read()  # 从摄像头捕获一帧
        if ret:
            self.current_frame = frame  # 保存当前帧
            self.display_image(frame)  # 显示当前帧

    def on_recognition_button_clicked(self):
        print("点击按钮了", self.is_recognizing)
        if not self.is_recognizing:
            self.start_recognition()  # 开始识别逻辑
        else:
            self.reset_void()

    def start_recognition(self):
        if self.current_frame is not None:
            self.is_recognizing = True  # 更新识别状态
            # 禁用按钮并显示加载提示
            self.button.setEnabled(False)
            self.display_loading()
            # 停止摄像头更新
            self.timer.stop()
            # 启动识别线程
            self.start_recognition_thread()

    def reset_void(self):
        print("finish_recognition")
        self.is_recognizing = False  # 更新识别状态
        self.button.setText('开始识别')  # 更新按钮文本为“重新识别”
        self.button.setEnabled(True)  # 重新启用按钮
        self.timer.start(20)  # 重新开始摄像头更新
        # 停止视频播放
        self.media_player.stop()
        self.video_widget.setVisible(False)  # 隐藏视频播放器
        self.image_label.setVisible(True)  # 显示图像标签

    def finish_recognition(self):
        print("finish_recognition")
        self.is_recognizing = True    # 更新识别状态
        self.button.setText('重新识别')  # 更新按钮文本为“重新识别”
        self.button.setEnabled(True)  # 重新启用按钮
        self.timer.start(20)  # 重新开始摄像头更新

    def start_recognition_thread(self):
        self.detect_stone(self.current_frame)
        # 在新的线程中进行识别
        #self.thread = RecognitionThread(self.current_frame, self.stone_colors, self.stone_texture)
        #self.thread.result_ready.connect(self.display_results)
        #self.thread.start()
        # 延迟3秒后显示结果
        #QTimer.singleShot(2000, self.show_results)

    def detect_stone(self, image):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        input_image = Image.fromarray(image)
        input_image = input_image.convert("RGB")
        input_image = input_image.resize((224, 224))
        img_chw = process_imageNdarray(input_image)
        if torch.cuda.is_available():
            img_chw = img_chw.view(1, 3, 224, 224).to(device)
        else:
            img_chw = img_chw.view(1, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            torch.no_grad()
            try:
                out = model(img_chw)
            except Exception as e:
                print(f"Error occurred: {e}")
            score = torch.nn.functional.softmax(out, dim=1)[0] * 100
            predicted = torch.max(out, 1)[1]
            score = score[predicted.item()].item()
            txt = str(classes[predicted.item()])

        print("结果是: ", str(txt))
        self.delay_show_result(str(txt))

    def show_results(self):
        if hasattr(self, 'thread'):
            # Check if thread is still running
            if self.thread.isRunning():
                self.thread.terminate()  # Terminate the thread if it's still running
            self.finish_recognition()  # Finish recognition process

    def display_results(self,  result_text):
        # 延迟3秒后显示结果
        QTimer.singleShot(2000, lambda: self.delay_show_result(result_text))
        #self.result_text.setPlainText(result_text)


    def delay_show_result(self, result_text):
        print("识别结果是", result_text)
        self.result_text.setPlainText(f'检测到的矿石类型: {result_text}')
        # 获取对应的视频序号，如果没有对应关系则使用默认序号

        video_index = self.video_mapping.get(result_text, "1")  # 默认序号为 "0"
        self.finish_recognition()  # Finish recognition process
        self.play_video(video_index)


    def display_image(self, img):
        # BGR转为RGB
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        self.capture.release()  # 释放摄像头
        event.accept()

    def handle_media_status(self, status):
        print(status)
        if status == QMediaPlayer.InvalidMedia:
            print("无效媒体")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = StoneRecognitionApp()
    window.show()
    sys.exit(app.exec_())

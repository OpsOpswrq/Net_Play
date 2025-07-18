import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QFileDialog, QWidget, QComboBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import torch
from AnimeGANV2 import AnimeGANV2
from PIL import Image

# 导入处理模块
from arcane_processor import process_image as process_arcane
from shinkai_processor import process_image as process_shinkai
from paprika_processor import process_image as process_paprika

camera_index = 1

class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_pixmap = None
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        
        # 检查CUDA可用性
        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_available else "cpu")
        self.device_options = ["CPU"]
        if self.cuda_available:
            self.device_options.append("CUDA")
        
        # 根据CUDA可用性动态设置style_dict
        self.style_dict = {
            "柏皮卡": "./weights/Shinkai.pt"
        }
        if self.cuda_available:
            self.style_dict["双城之战"] = "./weights/ArcaneGANv0.4.jit"
        
        # 加载模型并移动到默认设备
        self.models = {}
        try:
            for key, path in self.style_dict.items():
                if key == "双城之战":
                    model = torch.jit.load(path).to(self.device)
                    model = model.eval().half()  # 使用 FP16
                    for param in model.parameters():
                        param.data = param.data.to(self.device)
                    for buffer in model.buffers():
                        buffer.data = buffer.data.to(self.device)
                else:
                    model = torch.jit.load(path).to(self.device)
                    model = model.eval()
                self.models[key] = model
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            self.statusBar().showMessage(f"模型加载失败: {str(e)}")
            return
        
        self.initUI()

    def initUI(self):
        self.setWindowTitle('次元突破：让你身处动漫世界')
        self.setStyleSheet("background-color: #f5f5f5;")
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        self.image_label = QLabel('请选择一个图像文件或打开摄像头')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px solid #dcdcdc;
                border-radius: 8px;
                background-color: #ffffff;
                padding: 5px;
            }
        """)
        main_layout.addWidget(self.image_label, stretch=3)
        
        right_layout = QVBoxLayout()
        right_layout.setAlignment(Qt.AlignVCenter)
        right_layout.setSpacing(15)
        
        self.combo_box_style = QComboBox()
        self.combo_box_style.addItems(self.style_dict.keys())  # 动态添加样式选项
        self.combo_box_style.setFixedWidth(150)
        self.combo_box_style.setStyleSheet("""
            QComboBox {
                border: 1px solid #dcdcdc;
                border-radius: 5px;
                padding: 5px;
                background-color: #ffffff;
                font-size: 14px;
            }
            QComboBox:hover {
                border-color: #3b5998;
            }
            QComboBox::drop-down {
                border-left: 1px solid #dcdcdc;
                width: 20px;
            }
        """)
        right_layout.addWidget(self.combo_box_style)
        
        self.combo_box_device = QComboBox()
        self.combo_box_device.addItems(self.device_options)
        self.combo_box_device.setFixedWidth(150)
        self.combo_box_device.setStyleSheet("""
            QComboBox {
                border: 1px solid #dcdcdc;
                border-radius: 5px;
                padding: 5px;
                background-color: #ffffff;
                font-size: 14px;
            }
            QComboBox:hover {
                border-color: #3b5998;
            }
            QComboBox::drop-down {
                border-left: 1px solid #dcdcdc;
                width: 20px;
            }
        """)
        self.combo_box_device.currentTextChanged.connect(self.change_device)
        right_layout.addWidget(self.combo_box_device)
        
        self.select_button = QPushButton('选择图像文件')
        self.select_button.clicked.connect(self.select_image)
        self.select_button.setFixedWidth(150)
        self.select_button.setStyleSheet("""
            QPushButton {
                background-color: #3b5998;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #4c70ba;
            }
            QPushButton:pressed {
                background-color: #2a4373;
            }
        """)
        right_layout.addWidget(self.select_button)
        
        self.process_button = QPushButton('处理图像')
        self.process_button.clicked.connect(self.process_image)
        self.process_button.setFixedWidth(150)
        self.process_button.setStyleSheet("""
            QPushButton {
                background-color: #3b5998;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #4c70ba;
            }
            QPushButton:pressed {
                background-color: #2a4373;
            }
        """)
        right_layout.addWidget(self.process_button)
        
        self.camera_button = QPushButton('打开摄像头')
        self.camera_button.clicked.connect(self.toggle_camera)
        self.camera_button.setFixedWidth(150)
        self.camera_button.setStyleSheet("""
            QPushButton {
                background-color: #3b5998;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #4c70ba;
            }
            QPushButton:pressed {
                background-color: #2a4373;
            }
        """)
        right_layout.addWidget(self.camera_button)
        
        self.save_button = QPushButton('保存图像')
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setFixedWidth(150)
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #3b5998;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #4c70ba;
            }
            QPushButton:pressed {
                background-color: #2a4373;
            }
        """)
        right_layout.addWidget(self.save_button)
        
        main_layout.addLayout(right_layout, stretch=1)
        
        self.statusBar().showMessage('就绪')
        
    def change_device(self, device_name):
        try:
            new_device = torch.device("cuda" if device_name == "CUDA" else "cpu")
            if new_device != self.device:
                self.device = new_device
                # 更新style_dict以确保与当前设备一致
                self.style_dict = {
                    "柏皮卡": "./weights/Shinkai.pt"
                }
                if self.device.type == "cuda":
                    self.style_dict["双城之战"] = "./weights/ArcaneGANv0.4.jit"
                
                # 更新combo_box_style
                self.combo_box_style.clear()
                self.combo_box_style.addItems(self.style_dict.keys())
                
                # 重新加载模型
                self.models.clear()
                for key, path in self.style_dict.items():
                    if key == "双城之战":
                        self.models[key] = torch.jit.load(path).to(self.device)
                        self.models[key] = self.models[key].eval().half()
                        for param in self.models[key].parameters():
                            param.data = param.data.to(self.device)
                        for buffer in self.models[key].buffers():
                            buffer.data = buffer.data.to(self.device)
                    else:
                        self.models[key] = torch.jit.load(path).to(self.device)
                        self.models[key] = self.models[key].eval()
                        for param in self.models[key].parameters():
                            param.data = param.data.to(self.device)
                        for buffer in self.models[key].buffers():
                            buffer.data = buffer.data.to(self.device)
                self.statusBar().showMessage(f'已切换到设备: {device_name}')
        except Exception as e:
            self.statusBar().showMessage(f'设备切换失败: {str(e)}')
            self.combo_box_device.setCurrentText("CPU")
            self.device = torch.device("cpu")
            # 更新style_dict和combo_box_style
            self.style_dict = {
                "柏皮卡": "./weights/Shinkai.pt"
            }
            self.combo_box_style.clear()
            self.combo_box_style.addItems(self.style_dict.keys())
            # 重新加载模型
            self.models.clear()
            for key, path in self.style_dict.items():
                self.models[key] = torch.jit.load(path).to(self.device)
                self.models[key] = self.models[key].eval()
                for param in self.models[key].parameters():
                    param.data = param.data.to(self.device)
                for buffer in self.models[key].buffers():
                    buffer.data = buffer.data.to(self.device)
    
    def select_image(self):
        self.stop_camera()
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, '选择图像文件', '', '图像文件 (*.png *.jpg *.jpeg *.bmp *.gif);;所有文件 (*)', options=options)
        
        if file_name:
            self.load_image(file_name)
            
    def load_image(self, file_path):
        try:
            self.current_image_path = file_path
            self.pixmap = QPixmap(file_path)
            if self.pixmap.isNull():
                self.image_label.setText('无法加载图像')
                self.current_pixmap = None
                self.statusBar().showMessage('加载失败')
            else:
                self.current_pixmap = self.pixmap
                scaled_pixmap = self.current_pixmap.scaled(
                    self.image_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
                self.statusBar().showMessage(f'已加载: {file_path}')
                self.adjust_window_size()
        except Exception as e:
            self.image_label.setText(f'加载图像时出错: {str(e)}')
            self.current_pixmap = None
            self.statusBar().showMessage('加载失败')
            
    def adjust_window_size(self):
        if self.current_pixmap is None or self.current_pixmap.isNull():
            return
        
        pixmap_size = self.current_pixmap.size()
        img_width, img_height = pixmap_size.width(), pixmap_size.height()
        
        screen = QApplication.primaryScreen().availableGeometry()
        max_width = screen.width() * 0.9
        max_height = screen.height() * 0.9
        
        window_width = img_width + 200
        window_height = max(img_height, 400)
        
        window_width = min(window_width, max_width)
        window_height = min(window_height, max_height)
        
        window_width = max(window_width, 600)
        window_height = max(window_height, 400)
        
        self.resize(int(window_width), int(window_height))
        self.move(int((screen.width() - window_width) / 2), 
                 int((screen.height() - window_height) / 2))
        
    def toggle_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                self.image_label.setText('无法打开摄像头')
                self.statusBar().showMessage('摄像头打开失败')
                self.cap = None
                return
            
            self.current_pixmap = None
            self.pixmap = None
            self.timer.start(30)
            self.camera_button.setText('关闭摄像头')
            self.statusBar().showMessage('摄像头已打开')
            
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.current_pixmap = QPixmap(width, height)
            self.adjust_window_size()
        else:
            self.stop_camera()
            self.image_label.setText('请选择一个图像文件或打开摄像头')
            self.statusBar().showMessage('摄像头已关闭')
            
    def stop_camera(self):
        if self.cap is not None and self.cap.isOpened():
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.camera_button.setText('打开摄像头')
        
    def process_image(self):
        if hasattr(self, 'pixmap') and not self.pixmap.isNull():
            cv_image = self.qpixmap_to_cvimage(self.pixmap)
            if cv_image is not None:
                selected_option = self.combo_box_style.currentText()
                processed_image = self.apply_cartoon_effect(cv_image, selected_option)
                if processed_image is not None:
                    processed_pixmap = self.cvimage_to_qpixmap(processed_image)
                    if not processed_pixmap.isNull():
                        self.current_pixmap = processed_pixmap
                        scaled_pixmap = processed_pixmap.scaled(
                            self.image_label.size(), 
                            Qt.KeepAspectRatio, 
                            Qt.SmoothTransformation
                        )
                        self.image_label.setPixmap(scaled_pixmap)
                        self.statusBar().showMessage('图像处理完成')
                        self.adjust_window_size()
                    else:
                        self.image_label.setText('图像处理失败')
                        self.current_pixmap = None
                        self.statusBar().showMessage('图像处理失败')
                else:
                    self.image_label.setText('图像处理失败')
                    self.current_pixmap = None
                    self.statusBar().showMessage('图像处理失败')
    
    def update_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if camera_index == 0:
                    frame = np.fliplr(frame)
                
                selected_option = self.combo_box_style.currentText()
                processed_frame = self.apply_cartoon_effect(frame, selected_option)
                if processed_frame is not None:
                    processed_pixmap = self.cvimage_to_qpixmap(processed_frame)
                    if not processed_pixmap.isNull():
                        self.current_pixmap = processed_pixmap
                        scaled_pixmap = processed_pixmap.scaled(
                            self.image_label.size(), 
                            Qt.KeepAspectRatio, 
                            Qt.SmoothTransformation
                        )
                        self.image_label.setPixmap(scaled_pixmap)
                    else:
                        self.image_label.setText('视频帧处理失败')
                        self.statusBar().showMessage('视频帧处理失败')
                else:
                    self.image_label.setText('视频帧处理失败')
                    self.statusBar().showMessage('视频帧处理失败')
                    
    def apply_cartoon_effect(self, img, style):
        try:
            model = self.models.get(style)
            if model is None:
                return None
            
            if style == "双城之战":
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                processed_img = process_arcane(pil_img, model, self.device)
                if processed_img is not None:
                    return cv2.cvtColor(np.array(processed_img), cv2.COLOR_RGB2BGR)
            else:
                processed_img = process_shinkai(img, model, self.device)
                if processed_img is not None:
                    return processed_img
            return None
        except Exception as e:
            print(f"卡通化处理失败: {str(e)}")
            self.statusBar().showMessage(f'卡通化处理失败: {str(e)}')
            return None
    
    def save_image(self):
        if self.current_pixmap is None or self.current_pixmap.isNull():
            self.image_label.setText('没有可保存的图像')
            self.statusBar().showMessage('没有可保存的图像')
            return
        
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, '保存图像', '', '图像文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)', options=options)
        
        if file_name:
            try:
                if self.current_pixmap.save(file_name):
                    self.statusBar().showMessage(f'图像已保存到: {file_name}')
                else:
                    self.image_label.setText('图像保存失败')
                    self.statusBar().showMessage('图像保存失败')
            except Exception as e:
                self.image_label.setText(f'保存图像时出错: {str(e)}')
                self.statusBar().showMessage('保存失败')
    
    def qpixmap_to_cvimage(self, pixmap):
        qimage = pixmap.toImage()
        width = qimage.width()
        height = qimage.height()
        if qimage.format() in (QImage.Format_RGB32, QImage.Format_ARGB32):
            ptr = qimage.bits()
            ptr.setsize(height * width * 4)
            arr = np.frombuffer(ptr, np.uint8).reshape(height, width, 4)
            cv_image = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            return cv_image
        print(f"不支持的 QImage 格式: {qimage.format()}")
        return None
    
    def cvimage_to_qpixmap(self, cv_image):
        if len(cv_image.shape) == 3:
            height, width, channel = cv_image.shape
            bytes_per_line = 3 * width
            qimage = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        else:
            height, width = cv_image.shape
            bytes_per_line = width
            qimage = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimage)
            
    def resizeEvent(self, event):
        if self.current_pixmap is not None and not self.current_pixmap.isNull():
            scaled_pixmap = self.current_pixmap.scaled(
                self.image_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
        super().resizeEvent(event)
        
    def closeEvent(self, event):
        self.stop_camera()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageViewer()
    window.show()
    sys.exit(app.exec_())

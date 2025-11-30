import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                             QHBoxLayout, QWidget, QLabel, QCheckBox, QMessageBox,
                             QProgressBar)
from PyQt5.QtCore import Qt, QTimer, QThread
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np

from mouth_detector import MouthDetector
from video_thread import VideoThread


class MouthDetectionUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detector = MouthDetector()
        self.video_thread = None
        self.initUI()

        # 初始化最大位移变量
        self.max_open_distance = 0.0
        self.max_left_distance = 0.0  # 将存储负值
        self.max_right_distance = 0.0

        # 当前动作状态
        self.current_action = None
        self.reached_maximum = False

        # 尝试加载已保存的数据
        # self.load_max_distances()

    def initUI(self):
        self.setWindowTitle('口型检测系统')
        self.setGeometry(100, 100, 800, 600)

        # 创建中心部件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 创建左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        main_layout.addWidget(control_panel)

        # 创建右侧布局
        right_layout = QVBoxLayout()

        # 添加视频显示
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        right_layout.addWidget(self.video_label)

        # 添加进度条
        progress_layout = QVBoxLayout()

        # 垂直进度条（用于张口）
        self.vertical_progress = QProgressBar()
        self.vertical_progress.setMinimum(0)
        self.vertical_progress.setMaximum(1000)  # 提高精度
        self.vertical_progress.setOrientation(Qt.Vertical)  # 设置为垂直方向
        self.vertical_progress.setFixedHeight(200)  # 设置固定高度
        self.vertical_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: red;
            }
        """)
        # 水平进度条（用于左右运动）
        self.horizontal_progress = QProgressBar()
        self.horizontal_progress.setMinimum(0)
        self.horizontal_progress.setMaximum(1000)  # 提高精度
        self.horizontal_progress.setOrientation(Qt.Horizontal)  # 设置为水平方向
        self.horizontal_progress.setFixedWidth(200)  # 设置固定宽度
        self.horizontal_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: red;
            }
        """)

        # 创建进度条容器
        progress_container = QWidget()
        progress_container_layout = QHBoxLayout(progress_container)
        progress_container_layout.addWidget(self.vertical_progress)
        progress_container_layout.addWidget(self.horizontal_progress)

        self.progress_label = QLabel('当前值与最大值比例: 0%')
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(progress_container)

        right_layout.addLayout(progress_layout)
        main_layout.addLayout(right_layout)

        # 初始隐藏所有进度条
        self.vertical_progress.hide()
        self.horizontal_progress.hide()

        # 添加压力选择复选框
        self.pressure_checkbox = QCheckBox('是否施加压力')
        control_layout.addWidget(self.pressure_checkbox)

        # 添加校准按钮
        calibration_label = QLabel('校准测量:')
        control_layout.addWidget(calibration_label)

        self.open_button = QPushButton('张开距离')
        self.left_button = QPushButton('左侧运动')
        self.right_button = QPushButton('右侧运动')

        control_layout.addWidget(self.open_button)
        control_layout.addWidget(self.left_button)
        control_layout.addWidget(self.right_button)

        # 添加训练按钮
        training_label = QLabel('训练模式:')
        control_layout.addWidget(training_label)

        self.open_training_button = QPushButton('开始张口训练')
        self.left_training_button = QPushButton('开始左侧训练')
        self.right_training_button = QPushButton('开始右侧训练')

        control_layout.addWidget(self.open_training_button)
        control_layout.addWidget(self.left_training_button)
        control_layout.addWidget(self.right_training_button)

        # 添加状态显示
        self.status_label = QLabel('当前状态: 未开始检测')
        self.instruction_label = QLabel('请按照提示进行操作')
        self.maximum_label = QLabel('')  # 新增达到最大值的提示标签
        control_layout.addWidget(self.status_label)
        control_layout.addWidget(self.instruction_label)
        control_layout.addWidget(self.maximum_label)

        # 添加测量值显示
        self.measurement_label = QLabel('当前位移: 0.000')
        control_layout.addWidget(self.measurement_label)

        # 连接按钮信号
        self.open_button.clicked.connect(lambda: self.start_calibration('open'))
        self.left_button.clicked.connect(lambda: self.start_calibration('left'))
        self.right_button.clicked.connect(lambda: self.start_calibration('right'))

        # 连接训练按钮信号
        self.open_training_button.clicked.connect(lambda: self.start_training('open'))
        self.left_training_button.clicked.connect(lambda: self.start_training('left'))
        self.right_training_button.clicked.connect(lambda: self.start_training('right'))
        # 创建定时器用于更新提示
        self.instruction_timer = QTimer()
        self.instruction_timer.timeout.connect(self.update_instruction)
        self.current_instruction = 0

        # 初始化检测状态
        self.detection_running = False

    def update_progress_bar_style(self, progress_bar, progress):
        """更新进度条样式"""
        if progress >= 100:
            progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 2px solid grey;
                    border-radius: 5px;
                    text-align: center;
                    background-color: #f0f0f0;
                }
                QProgressBar::chunk {
                    background-color: #4CAF50;
                }
            """)
        else:
            progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 2px solid grey;
                    border-radius: 5px;
                    text-align: center;
                    background-color: #f0f0f0;
                }
                QProgressBar::chunk {
                    background-color: red;
                }
            """)

    def start_calibration(self, mode):
        """开始校准过程"""
        if self.video_thread is not None:
            self.stop_detection()

        self.detector.reset_calibration()
        self.detector.calibration_mode = mode
        self.current_action = mode

        # 只重置当前校准模式对应的值
        if mode == 'left':
            self.max_left_distance = 0.0
        elif mode == 'right':
            self.max_right_distance = 0.0
        elif mode == 'open':
            self.max_open_distance = 0.0

        self.status_label.setText(f'正在校准{self.get_mode_name(mode)}...')
        self.maximum_label.setText('')

        # 校准模式下隐藏进度条
        self.vertical_progress.hide()
        self.horizontal_progress.hide()
        self.progress_label.setText('当前值与最大值比例: 0%')

        # 启动视频线程
        self.video_thread = VideoThread(self.detector)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.measurement_signal.connect(self.update_measurement)
        self.video_thread.start()

    def start_training(self, mode):
        """开始特定模式的训练"""
        if self.video_thread is not None:
            self.stop_detection()

        self.detection_running = True
        self.current_action = mode
        self.status_label.setText(f'正在进行{self.get_mode_name(mode)}训练...')
        self.maximum_label.setText('')

        # 重置进度条
        self.vertical_progress.setValue(0)
        self.horizontal_progress.setValue(0)
        self.progress_label.setText('当前值与最大值比例: 0%')

        # 设置对应模式的训练指令序列
        if mode == 'open':
            self.instructions = [
                                    ('rest', '1. 自然闭口位'),
                                    ('open', '2. 缓慢张口'),
                                    ('open', '3. 最大开口位保持1-2秒'),
                                    ('rest', '4. 缓慢返回至自然闭口位')
                                ] * 8  # 重复8次
        elif mode == 'left':
            self.instructions = [
                                    ('rest', '1. 自然闭口位'),
                                    ('left', '2. 缓慢向左侧运动'),
                                    ('left', '3. 最大左侧位保持1-2秒'),
                                    ('rest', '4. 缓慢返回至自然闭口位')
                                ] * 8  # 重复8次
        elif mode == 'right':
            self.instructions = [
                                    ('rest', '1. 自然闭口位'),
                                    ('right', '2. 缓慢向右侧运动'),
                                    ('right', '3. 最大右侧位保持1-2秒'),
                                    ('rest', '4. 缓慢返回至自然闭口位')
                                ] * 8  # 重复8次
        self.current_instruction = 0
        self.update_current_instruction()

        # 启动视频线程
        self.video_thread = VideoThread(self.detector)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.measurement_signal.connect(self.update_measurement)
        self.video_thread.start()

        # 启动指令定时器
        self.instruction_timer.start(5000)  # 每5秒更新一次指令

    def get_mode_name(self, mode):
        """获取模式的中文名称"""
        mode_names = {
            'open': '张口',
            'left': '左侧',
            'right': '右侧'
        }
        return mode_names.get(mode, mode)

    def load_max_distances(self):
        """从文件加载最大位移数据"""
        try:
            with open('max_distances.txt', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if '最大张嘴位移:' in line:
                        self.max_open_distance = float(line.split(':')[1])
                    elif '最大左侧位移:' in line:
                        self.max_left_distance = float(line.split(':')[1])
                    elif '最大右侧位移:' in line:
                        self.max_right_distance = float(line.split(':')[1])
        except FileNotFoundError:
            # 如果文件不存在，使用默认值
            self.max_open_distance = 0.0
            self.max_left_distance = 0.0
            self.max_right_distance = 0.0

    def update_current_instruction(self):
        """更新当前指令和动作类型"""
        if self.current_instruction < len(self.instructions):
            self.current_action, instruction_text = self.instructions[self.current_instruction]
            self.instruction_label.setText(instruction_text)
            self.reached_maximum = False
            self.maximum_label.setText('')
        else:
            # 训练完成
            self.stop_detection()
            self.instruction_label.setText('训练完成！')

    def update_instruction(self):
        """更新指令显示"""
        self.current_instruction = self.current_instruction + 1
        if self.current_instruction < len(self.instructions):
            self.update_current_instruction()
        else:
            # 训练完成
            self.stop_detection()
            self.instruction_label.setText('训练完成！')

    def update_image(self, frame):
        """更新视频显示"""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_image = convert_to_qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(scaled_image))

    def update_measurement(self, measurements):
        """更新测量值显示并保存最大位移"""
        if measurements:
            # 只在训练模式下显示进度条
            is_training = hasattr(self, 'instructions') and self.current_instruction < len(self.instructions)

            if is_training:
                current_value = 0
                max_value = 0

                if self.current_action == 'open' and "vertical" in measurements:
                    current_value = measurements["vertical"]
                    max_value = self.max_open_distance
                    # 显示垂直进度条，隐藏水平进度条
                    self.vertical_progress.show()
                    self.horizontal_progress.hide()
                    active_progress_bar = self.vertical_progress
                elif (self.current_action in ['left', 'right']) and "horizontal" in measurements:
                    current_value = abs(measurements["horizontal"])
                    max_value = abs(
                        self.max_left_distance if self.current_action == 'left' else self.max_right_distance)
                    # 显示水平进度条，隐藏垂直进度条
                    self.horizontal_progress.show()
                    self.vertical_progress.hide()
                    active_progress_bar = self.horizontal_progress

                # 更新进度条
                if max_value > 0:
                    progress = (current_value / max_value) * 1000  # 使用更精确的比例
                    active_progress_bar.setValue(min(int(progress), 1000))
                    percentage = (current_value / max_value) * 100
                    self.progress_label.setText(f'当前值与最大值比例: {percentage:.1f}%')
                    self.update_progress_bar_style(active_progress_bar, percentage)
            else:
                # 非训练模式下隐藏进度条
                self.vertical_progress.hide()
                self.horizontal_progress.hide()
                self.progress_labeQAl.setText('当前值与最大值比例: 0%')

                # 只在非训练模式下更新最大位移数据
                # 处理垂直方向（开口）的测量
                if "vertical" in measurements and self.current_action == 'open':
                    current_value = measurements["vertical"]
                    if current_value > 0:  # 只处理正值
                        if current_value > self.max_open_distance:
                            self.max_open_distance = current_value
                            self.check_maximum(current_value, self.max_open_distance)
                            self.save_max_distances()

                # 处理水平方向（左右）的测量
                if "horizontal" in measurements:
                    current_value = measurements["horizontal"]

                    # 左侧运动（负值）
                    if self.current_action == 'left':
                        if current_value > 0:  # 只处理负值
                            if self.max_left_distance == 0 or current_value > self.max_left_distance:
                                self.max_left_distance = current_value
                                self.check_maximum(abs(current_value), abs(self.max_left_distance))
                                self.save_max_distances()

                    # 右侧运动（正值）
                    elif self.current_action == 'right':
                        if current_value > 0:  # 只处理正值
                            if current_value > self.max_right_distance:
                                self.max_right_distance = current_value
                                self.check_maximum(current_value, self.max_right_distance)
                                self.save_max_distances()

            # 更新显示
            self.measurement_label.setText(
                f'当前位移: {measurements.get("displacement", 0):.3f}\n'
                f'最大张开: {self.max_open_distance:.3f}\n'
                f'最大左侧: {self.max_left_distance:.3f}\n'
                f'最大右侧: {self.max_right_distance:.3f}'
            )

    def check_maximum(self, current_value, max_value):
        """检查是否达到最大值"""
        if not self.reached_maximum and max_value != 0:  # 添加对0的检查
            threshold = 0.9  # 设定阈值为最大值的90%
            if current_value >= max_value * threshold:
                self.maximum_label.setText('已达到最大值！')
                self.reached_maximum = True
            else:
                self.maximum_label.setText('未达到最大值')

    def stop_detection(self):
        """停止检测并保存最大位移"""
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread = None

        self.detection_running = False
        self.status_label.setText('检测已停止')
        self.instruction_timer.stop()
        self.video_label.clear()
        self.maximum_label.setText('')

        # 隐藏进度条
        self.vertical_progress.hide()
        self.horizontal_progress.hide()
        self.progress_label.setText('当前值与最大值比例: 0%')

        # 保存最大位移数据
        self.save_max_distances()

    def save_max_distances(self):
        """保存最大位移数据"""
        with open('max_distances.txt', 'w') as f:
            f.write(f'最大张嘴位移: {self.max_open_distance:.3f}\n')
            f.write(f'最大左侧位移: {self.max_left_distance:.3f}\n')  # 保存负值
            f.write(f'最大右侧位移: {self.max_right_distance:.3f}\n')

    def closeEvent(self, event):
        """程序关闭时的清理工作"""
        self.stop_detection()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MouthDetectionUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
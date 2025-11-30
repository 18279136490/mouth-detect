from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, \
    QProgressBar
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import sys
import cv2
import numpy as np
from mouth_detector import MouthDetector


class MouthDetectionUI(QMainWindow):
    def __init__(self, mouth_detector):
        super().__init__()
        self.detector = mouth_detector
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Mouth Detection Training')
        self.setGeometry(100, 100, 800, 600)

        # Layouts
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left Control Panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        main_layout.addWidget(control_panel)

        # Right Layout for Video and Progress Bar
        right_layout = QVBoxLayout()

        # Video Display Area
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        right_layout.addWidget(self.video_label)

        # Progress Bars
        self.vertical_progress = QProgressBar()
        self.vertical_progress.setOrientation(Qt.Vertical)
        self.vertical_progress.setMinimum(0)
        self.vertical_progress.setMaximum(1000)
        right_layout.addWidget(self.vertical_progress)

        self.horizontal_progress = QProgressBar()
        self.horizontal_progress.setOrientation(Qt.Horizontal)
        self.horizontal_progress.setMinimum(0)
        self.horizontal_progress.setMaximum(1000)
        right_layout.addWidget(self.horizontal_progress)

        # Add Right Layout to the main layout
        main_layout.addLayout(right_layout)

        # Action instructions and current state
        self.instruction_label = QLabel('Please follow the instructions.')
        control_layout.addWidget(self.instruction_label)

        # Calibration Results
        self.calibration_label = QLabel('Calibration Results: Max Open: 0.0')
        control_layout.addWidget(self.calibration_label)

        # Connect buttons
        self.start_button = QPushButton('Start Training')
        control_layout.addWidget(self.start_button)
        self.start_button.clicked.connect(self.start_training)

        # Timer to update instructions
        self.instruction_timer = QTimer()
        self.instruction_timer.timeout.connect(self.update_instruction)

        # Initialize detection
        self.detection_running = False

    def update_image(self, frame):
        """Update video display"""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_image = convert_to_qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(scaled_image))

    def update_measurements(self, measurements):
        """Update displacement and measurement values"""
        # Update displacement, vertical, and horizontal readings
        displacement = measurements['displacement']
        vertical = measurements['vertical']
        horizontal = measurements['horizontal']

        self.calibration_label.setText(
            f'Displacement: {displacement:.3f} | Vertical: {vertical:.3f} | Horizontal: {horizontal:.3f}')

        # Update progress bars
        self.vertical_progress.setValue(int((vertical / self.detector.max_open) * 1000))
        self.horizontal_progress.setValue(int((horizontal / self.detector.max_right) * 1000))

    def start_training(self):
        """Start training (in this case, mouth detection and calibration)"""
        self.detection_running = True
        self.instruction_label.setText('Training started... Follow the instructions.')

        # Start video processing and measurement updates
        # Implement video feed capturing, measurement extraction, and updates
        # You would typically use a thread or another mechanism to periodically call the update_measurements function with real-time data

    def update_instruction(self):
        """Update the instructions displayed for the user"""
        self.instruction_label.setText('Next step: Slowly open your mouth')

    def closeEvent(self, event):
        """Close the application and stop detection"""
        self.detection_running = False
        event.accept()


def main():
    # Initialize the mouth detector
    mouth_detector = MouthDetector()  # Ensure you have the MouthDetector class defined or imported

    # Create the PyQt5 application
    app = QApplication(sys.argv)
    window = MouthDetectionUI(mouth_detector)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

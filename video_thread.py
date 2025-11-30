from PyQt5.QtCore import QThread, pyqtSignal
import cv2
import numpy as np

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    measurement_signal = pyqtSignal(dict)  # 修改为发送字典类型的数据

    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if ret:
                measurement = self.detector.process_frame(frame)
                if measurement is not None:
                    self.measurement_signal.emit(measurement)
                self.change_pixmap_signal.emit(frame)
        cap.release()

    def stop(self):
        self.running = False
        self.wait()
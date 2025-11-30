import cv2
import mediapipe as mp
import numpy as np
import time


class MouthDetector:
    def __init__(self):
        """初始化嘴部检测器"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.drawing.DrawingSpec(thickness=1, circle_radius=1)

        # 初始化测量值
        self.max_open = 0
        self.max_left = 0
        self.max_right = 0
        self.initial_position = None
        self.calibration_mode = None

        # 定义关键点索引
        self.MOUTH_POINTS = {
            'top_lip': 13,  # 上嘴唇中点
            'bottom_lip': 14,  # 下嘴唇中点
            'left_corner': 78,  # 左嘴角
            'right_corner': 308,  # 右嘴角
            'middle_lower': 17,  # 下嘴唇中点（用于位移计算）
            'left_top': 76,  # 左上嘴唇
            'right_top': 306,  # 右上嘴唇
            'left_bottom': 77,  # 左下嘴唇
            'right_bottom': 307  # 右下嘴唇
        }

        # 初始化CSV文件
        self.frame_count = 0
        self.measurements_history = []

        # 添加动作状态跟踪
        self.action_state = 'neutral'  # 可能的状态：neutral, open, left, right
        self.action_start_time = 0  # 动作开始时间
        self.current_action_duration = 0  # 当前动作持续时间
        self.last_position = None  # 上一帧的位置
        self.last_time = None  # 上一帧的时间

        # 动作阈值
        self.OPEN_THRESHOLD = 0.1  # 张嘴阈值
        self.MOVEMENT_THRESHOLD = 0.05  # 左右移动阈值

        # 动作统计
        self.action_stats = {
            'open': {'total_time': 0, 'count': 0, 'avg_speed': 0},
            'left': {'total_time': 0, 'count': 0, 'avg_speed': 0},
            'right': {'total_time': 0, 'count': 0, 'avg_speed': 0}
        }

    def detect_action(self, vertical_dist, displacement, current_time):
        """检测当前动作并计算持续时间和速度"""
        # 确定当前动作
        new_state = 'neutral'
        if vertical_dist > self.OPEN_THRESHOLD:
            new_state = 'open'
        elif displacement < -self.MOVEMENT_THRESHOLD:
            new_state = 'left'
        elif displacement > self.MOVEMENT_THRESHOLD:
            new_state = 'right'

        # 如果是新动作
        if new_state != self.action_state:
            if self.action_state != 'neutral':
                # 记录上一个动作的统计信息
                duration = current_time - self.action_start_time
                self.action_stats[self.action_state]['total_time'] += duration
                self.action_stats[self.action_state]['count'] += 1

            # 更新状态
            self.action_state = new_state
            self.action_start_time = current_time

        # 计算当前动作持续时间
        if self.action_state != 'neutral':
            self.current_action_duration = current_time - self.action_start_time

    def calculate_speed(self, current_position, current_time):
        """计算动作速度"""
        if self.last_position is not None and self.last_time is not None:
            time_diff = current_time - self.last_time
            if time_diff > 0:
                distance = np.linalg.norm(current_position - self.last_position)
                speed = distance / time_diff

                # 更新动作的平均速度
                if self.action_state != 'neutral':
                    stats = self.action_stats[self.action_state]
                    stats['avg_speed'] = (stats['avg_speed'] * stats['count'] + speed) / (stats['count'] + 1)

        # 更新上一帧的位置和时间
        self.last_position = current_position.copy()
        self.last_time = current_time

    def calculate_mouth_distances(self, landmarks):
        """计算嘴部各种距离"""
        # 获取关键点
        top_lip = np.array([landmarks[self.MOUTH_POINTS['top_lip']].x,
                            landmarks[self.MOUTH_POINTS['top_lip']].y])
        bottom_lip = np.array([landmarks[self.MOUTH_POINTS['bottom_lip']].x,
                               landmarks[self.MOUTH_POINTS['bottom_lip']].y])
        left_corner = np.array([landmarks[self.MOUTH_POINTS['left_corner']].x,
                                landmarks[self.MOUTH_POINTS['left_corner']].y])
        right_corner = np.array([landmarks[self.MOUTH_POINTS['right_corner']].x,
                                 landmarks[self.MOUTH_POINTS['right_corner']].y])

        # 计算垂直张开距离
        vertical_distance = np.linalg.norm(top_lip - bottom_lip)

        # 计算水平距离（左右嘴角之间的距离）
        horizontal_distance = np.linalg.norm(left_corner - right_corner)

        # 计算旋转（使用上下嘴唇中点）
        mouth_center_vertical = (top_lip + bottom_lip) / 2
        left_rotation = np.linalg.norm(top_lip - mouth_center_vertical)
        right_rotation = np.linalg.norm(bottom_lip - mouth_center_vertical)

        return vertical_distance, horizontal_distance, left_rotation, right_rotation

    def process_frame(self, frame):
        """处理视频帧"""
        current_time = time.time()  # 获取当前时间
        self.frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # 绘制面部网格
            self.drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=self.drawing_spec,
                connection_drawing_spec=self.drawing_spec)

            # 获取上嘴唇中点位置
            upper_lip = np.array([face_landmarks.landmark[self.MOUTH_POINTS['top_lip']].x,
                                  face_landmarks.landmark[self.MOUTH_POINTS['top_lip']].y])

            # 计算嘴部距离
            vertical_dist, horizontal_dist, left_rot, right_rot = self.calculate_mouth_distances(
                face_landmarks.landmark)

            # 如果是第一帧，初始化初始位置
            if self.initial_position is None:
                self.initial_position = upper_lip.copy()
                return {
                    'frame': self.frame_count,
                    'displacement': 0,
                    'vertical': vertical_dist,
                    'horizontal': horizontal_dist,
                    'left_rotation': left_rot,
                    'right_rotation': right_rot
                }

            # 计算水平位移
            displacement = upper_lip[0] - self.initial_position[0]

            # 检测动作和计算速度
            self.detect_action(vertical_dist, displacement, current_time)
            self.calculate_speed(upper_lip, current_time)

            # 如果在校准模式下，更新最大值
            if self.calibration_mode == 'open':
                self.max_open = max(self.max_open, vertical_dist)
            elif self.calibration_mode == 'left':
                self.max_left = min(self.max_left, displacement)
            elif self.calibration_mode == 'right':
                self.max_right = max(self.max_right, displacement)

            # 在图像上绘制测量点和位移线
            self.draw_measurements(frame, face_landmarks.landmark, displacement,
                                   vertical_dist, horizontal_dist, left_rot, right_rot)

            # 保存测量结果
            measurements = {
                'frame': self.frame_count,
                'displacement': displacement,
                'vertical': vertical_dist,
                'horizontal': horizontal_dist,
                'left_rotation': left_rot,
                'right_rotation': right_rot
            }
            self.measurements_history.append(measurements)

            return measurements
        return None

    def draw_measurements(self, frame, landmarks, displacement, vertical_dist,
                          horizontal_dist, left_rot, right_rot):
        """绘制测量结果"""
        h, w = frame.shape[:2]

        # 获取上下嘴唇中点的当前位置
        top_x = int(landmarks[self.MOUTH_POINTS['top_lip']].x * w)
        top_y = int(landmarks[self.MOUTH_POINTS['top_lip']].y * h)
        bottom_x = int(landmarks[self.MOUTH_POINTS['bottom_lip']].x * w)
        bottom_y = int(landmarks[self.MOUTH_POINTS['bottom_lip']].y * h)

        # 绘制跟踪点
        cv2.circle(frame, (top_x, top_y), 3, (0, 0, 255), -1)  # 红色点跟踪上嘴唇
        cv2.circle(frame, (bottom_x, bottom_y), 3, (255, 0, 0), -1)  # 蓝色点跟踪下嘴唇

        # 获取并绘制嘴部关键点
        top_lip = (top_x, top_y)
        bottom_lip = (bottom_x, bottom_y)
        left_corner = (int(landmarks[self.MOUTH_POINTS['left_corner']].x * w),
                       int(landmarks[self.MOUTH_POINTS['left_corner']].y * h))
        right_corner = (int(landmarks[self.MOUTH_POINTS['right_corner']].x * w),
                        int(landmarks[self.MOUTH_POINTS['right_corner']].y * h))

        # 绘制上下嘴唇中点和连接线
        cv2.circle(frame, top_lip, 2, (0, 255, 255), -1)  # 黄色点标记上嘴唇中点
        cv2.circle(frame, bottom_lip, 2, (0, 255, 255), -1)  # 黄色点标记下嘴唇中点
        cv2.line(frame, top_lip, bottom_lip, (0, 255, 0), 2)  # 绿色线连接上下嘴唇中点

        # 绘制嘴角连接线
        cv2.line(frame, left_corner, right_corner, (0, 255, 0), 2)  # 水平线（绿色）

        # 显示测量值
        cv2.putText(frame, f'Displacement: {displacement:.3f}', (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f'Vertical: {vertical_dist:.3f}', (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Horizontal: {horizontal_dist:.3f}', (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Left Rot: {left_rot:.3f}', (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Right Rot: {right_rot:.3f}', (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 显示当前动作信息
        if self.action_state != 'neutral':
            cv2.putText(frame, f'Action: {self.action_state}', (30, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f'Duration: {self.current_action_duration:.2f}s', (30, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 显示当前动作的统计信息
            stats = self.action_stats[self.action_state]
            cv2.putText(frame, f'Avg Speed: {stats["avg_speed"]:.3f}', (30, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f'Total Time: {stats["total_time"]:.2f}s', (30, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 如果在校准模式下，显示最大值
        if self.calibration_mode:
            if self.calibration_mode == 'open':
                cv2.putText(frame, f'Max Open: {self.max_open:.3f}', (30, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif self.calibration_mode == 'left':
                cv2.putText(frame, f'Max Left: {self.max_left:.3f}', (30, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif self.calibration_mode == 'right':
                cv2.putText(frame, f'Max Right: {self.max_right:.3f}', (30, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def reset_calibration(self):
        """重置校准数据"""
        self.initial_position = None
        self.calibration_mode = None
        self.max_left = 0
        self.max_right = 0
        self.max_open = 0
        self.frame_count = 0
        self.measurements_history = []

        # 重置动作统计
        self.action_state = 'neutral'
        self.action_start_time = 0
        self.current_action_duration = 0
        self.last_position = None
        self.last_time = None
        self.action_stats = {
            'open': {'total_time': 0, 'count': 0, 'avg_speed': 0},
            'left': {'total_time': 0, 'count': 0, 'avg_speed': 0},
            'right': {'total_time': 0, 'count': 0, 'avg_speed': 0}
        }

    def get_calibration_results(self):
        """获取校准结果"""
        return {
            'max_open': self.max_open,
            'max_left': self.max_left,
            'max_right': self.max_right
        }

    def get_measurements_history(self):
        """获取测量历史数据"""
        return self.measurements_history

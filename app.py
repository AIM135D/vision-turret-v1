import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify
from ultralytics import YOLO

# PCA9685 servo driver
import board
import busio
from adafruit_pca9685 import PCA9685

app = Flask(__name__)

# ===================== System configuration =====================

CAM_INDEX = 0
FRAME_W, FRAME_H = 320, 240

STREAM_FPS_LIMIT = 15
JPEG_QUALITY = 55

MODEL_PATH = "yolov8n.pt"
CONF = 0.25
IMGSZ = 224
DETECT_HZ = 1

TRACKER_TYPE = "KCF"
TRACK_HZ = 30

TRACK_CLASS_ID = 0  # COCO: person

# PD control parameters
DEADBAND = 6
Kp_x, Kp_y = 0.10, 0.10
Kd_x, Kd_y = 0.18, 0.18

LPF_ALPHA = 0.50
MAX_STEP = 120

LOST_HOLD_SEC = 0.50

# ===================== Servo configuration =====================

SERVO_USE_REAL = True
PITCH_ENABLED = True

YAW_CH = 8
YAW_MIN_US, YAW_MAX_US, YAW_CENTER_US = 1000, 2050, 1500

PITCH_CH = 7
PITCH_MIN_US, PITCH_MAX_US, PITCH_CENTER_US = 950, 2100, 1500

INVERT_YAW = False
INVERT_PITCH = False

SERVO_MAX_US_STEP_PER_UPDATE = 24


# ===================== Servo driver =====================

class ServoPCA9685:

    def __init__(self):

        self.yaw = YAW_CENTER_US
        self.pitch = PITCH_CENTER_US
        self.last_print = 0.0

        i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(i2c)
        self.pca.frequency = 50

        self._write_us(YAW_CH, self.yaw)
        self._write_us(PITCH_CH, self.pitch)

    @staticmethod
    def _clamp(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    @staticmethod
    def _us_to_duty(us: int) -> int:
        return int(us * 65535 / 20000)

    def _write_us(self, ch: int, us: int):
        self.pca.channels[ch].duty_cycle = self._us_to_duty(us)

    @staticmethod
    def _limit_step(cur: int, delta: int, max_step: int) -> int:

        if delta > max_step:
            delta = max_step
        elif delta < -max_step:
            delta = -max_step

        return cur + delta

    def update(self, dyaw: int, dpitch: int):

        if INVERT_YAW:
            dyaw = -dyaw

        if INVERT_PITCH:
            dpitch = -dpitch

        if not PITCH_ENABLED:
            dpitch = 0

        self.yaw = self._limit_step(self.yaw, dyaw, SERVO_MAX_US_STEP_PER_UPDATE)
        self.pitch = self._limit_step(self.pitch, dpitch, SERVO_MAX_US_STEP_PER_UPDATE)

        self.yaw = int(self._clamp(self.yaw, YAW_MIN_US, YAW_MAX_US))
        self.pitch = int(self._clamp(self.pitch, PITCH_MIN_US, PITCH_MAX_US))

        try:
            self._write_us(YAW_CH, self.yaw)
            self._write_us(PITCH_CH, self.pitch)

        except Exception as e:
            print("Servo I2C error:", e)

        now = time.time()

        if now - self.last_print > 0.25:
            print(f"yaw={self.yaw} pitch={self.pitch}")
            self.last_print = now


class ServoMock:

    def __init__(self):

        self.yaw = 1500
        self.pitch = 1500

    def update(self, dyaw: int, dpitch: int):

        self.yaw = int(np.clip(self.yaw + dyaw, 500, 2500))
        self.pitch = int(np.clip(self.pitch + dpitch, 500, 2500))


servo = ServoPCA9685() if SERVO_USE_REAL else ServoMock()


# ===================== PD controller =====================

class PDController:

    def __init__(self):

        self.dx_f = 0.0
        self.dy_f = 0.0
        self.prev_dx = 0.0
        self.prev_dy = 0.0
        self.prev_t = time.time()

    def step(self, dx: float, dy: float) -> Tuple[int, int]:

        if abs(dx) < DEADBAND:
            dx = 0.0

        if abs(dy) < DEADBAND:
            dy = 0.0

        self.dx_f = LPF_ALPHA * self.dx_f + (1.0 - LPF_ALPHA) * dx
        self.dy_f = LPF_ALPHA * self.dy_f + (1.0 - LPF_ALPHA) * dy

        now = time.time()
        dt = max(1e-3, now - self.prev_t)

        ddx = (self.dx_f - self.prev_dx) / dt
        ddy = (self.dy_f - self.prev_dy) / dt

        self.prev_dx = self.dx_f
        self.prev_dy = self.dy_f
        self.prev_t = now

        dyaw = int(-(Kp_x * self.dx_f + Kd_x * ddx))
        dpitch = int(-(Kp_y * self.dy_f + Kd_y * ddy))

        return dyaw, dpitch


controller = PDController()


# ===================== Camera =====================

cap = cv2.VideoCapture(CAM_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

try:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except:
    pass

if not cap.isOpened():
    raise RuntimeError("Camera open failed")

frame_lock = threading.Lock()

latest_frame = None
latest_frame_ts = 0.0


def camera_reader():

    global latest_frame, latest_frame_ts

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        with frame_lock:

            latest_frame = frame
            latest_frame_ts = time.time()


# ===================== Tracking state =====================

@dataclass
class TrackState:

    mode: str = "IDLE"
    bbox_xywh: Optional[Tuple[int, int, int, int]] = None
    target_center: Optional[Tuple[float, float]] = None
    dxdy: Optional[Tuple[float, float]] = None
    last_update_ts: float = 0.0


track_state = TrackState()


# ===================== YOLO =====================

model = YOLO(MODEL_PATH)


def detector_worker():

    global track_state

    while True:

        with frame_lock:

            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            continue

        results = model.predict(frame, conf=CONF, imgsz=IMGSZ, verbose=False)

        if len(results[0].boxes) == 0:
            continue

        box = results[0].boxes.xyxy[0].cpu().numpy()

        x1, y1, x2, y2 = box

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        track_state.mode = "TRK"
        track_state.target_center = (cx, cy)
        track_state.last_update_ts = time.time()


# ===================== Tracking control =====================

def tracker_worker():

    global track_state

    while True:

        if track_state.target_center is None:
            continue

        cx, cy = track_state.target_center

        dx = cx - FRAME_W / 2
        dy = cy - FRAME_H / 2

        dyaw, dpitch = controller.step(dx, dy)

        servo.update(dyaw, dpitch)


# ===================== MJPEG stream =====================

def gen_mjpeg():

    while True:

        with frame_lock:

            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            continue

        ok, buf = cv2.imencode(".jpg", frame)

        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(gen_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ===================== Main =====================

def main():

    threading.Thread(target=camera_reader, daemon=True).start()
    threading.Thread(target=detector_worker, daemon=True).start()
    threading.Thread(target=tracker_worker, daemon=True).start()

    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()

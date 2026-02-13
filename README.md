# Vision Turret v1

Embedded real-time vision tracking turret system based on Raspberry Pi 4B using YOLOv8, OpenCV Tracker, and PCA9685 servo control.

## Features

- Real-time object detection using YOLOv8
- High-frequency tracking using OpenCV KCF tracker
- PD closed-loop control for servo gimbal
- PCA9685 servo driver support
- Flask MJPEG live video stream
- Low latency optimized pipeline

## Hardware

- Raspberry Pi 4B (4GB)
- PCA9685 servo driver
- Servo motors (yaw / pitch)
- USB or CSI camera
- External 5V–6V servo power supply

## Installation

Create virtual environment:

python3 -m venv venv
source venv/bin/activate
Install dependencies:

pip install -r requirements.txt
Run:

python3 app.py
Open browser:

http://<raspberrypi_ip>:5000/video_feed
## License

MIT License

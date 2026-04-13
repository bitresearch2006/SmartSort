#!/usr/bin/env python3

import RPi.GPIO as GPIO
from gpiozero import Servo
from picamera2 import Picamera2
import time
import os
from datetime import datetime

# ---------------- GPIO SETUP ----------------
GPIO.setmode(GPIO.BCM)

# IR Sensor
IR_PIN = 17
GPIO.setup(IR_PIN, GPIO.IN)

# Ultrasonic
TRIG = 23
ECHO = 24
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# ---------------- SERVO SETUP ----------------
# Servo 1 → Bin selector (GPIO 12)
SERVO_BIN_PIN = 12
servo_bin = Servo(SERVO_BIN_PIN, min_pulse_width=0.0005, max_pulse_width=0.0025)

# Servo 2 → Lid control (GPIO 13)
SERVO_LID_PIN = 13
servo_lid = Servo(SERVO_LID_PIN, min_pulse_width=0.0005, max_pulse_width=0.0025)

# ---------------- CAMERA SETUP ----------------
SAVE_DIR = "captured_images"
os.makedirs(SAVE_DIR, exist_ok=True)

cam = Picamera2()
config = cam.create_still_configuration(main={"size": (320, 240)})
cam.configure(config)
cam.start()
time.sleep(2)

# ---------------- FUNCTIONS ----------------

def read_ir():
    return GPIO.input(IR_PIN) == 0  # True = object detected


def get_distance():
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    start = time.time()
    stop = time.time()

    while GPIO.input(ECHO) == 0:
        start = time.time()

    while GPIO.input(ECHO) == 1:
        stop = time.time()

    elapsed = stop - start
    distance = (elapsed * 34300) / 2
    return distance


def capture_image():
    filename = datetime.now().strftime("capture_%Y%m%d_%H%M%S.jpg")
    path = os.path.join(SAVE_DIR, filename)

    cam.capture_file(path)
    print(f"📸 Image saved: {path}")

    with open(path, "rb") as f:
        return f.read()


def set_servo_angle(servo, angle):
    value = (angle - 90) / 90
    servo.value = value


# Bin servo control
BIN_ANGLES = {
    1: 0,
    2: 45,
    3: 90,
    4: 135,
    5: 180
}

def move_bin(bin_number):
    angle = BIN_ANGLES.get(bin_number, 90)
    print(f"🧭 Moving bin servo → Bin {bin_number} ({angle}°)")
    set_servo_angle(servo_bin, angle)
    time.sleep(1)


# Lid control
def open_lid():
    print("🟢 Opening lid")
    set_servo_angle(servo_lid, 0)


def close_lid():
    print("🔴 Closing lid")
    set_servo_angle(servo_lid, 90)


def cleanup():
    cam.stop()
    servo_bin.detach()
    servo_lid.detach()
    GPIO.cleanup()

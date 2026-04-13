#!/usr/bin/env python3
import RPi.GPIO as GPIO
import time

IR_PIN = 17  # change if needed

GPIO.setmode(GPIO.BCM)
GPIO.setup(IR_PIN, GPIO.IN)

print("IR Sensor Test Started...")

try:
    while True:
        state = GPIO.input(IR_PIN)

        if state == 0:
            print("🔴 Object Detected")
        else:
            print("🟢 No Object")

        time.sleep(0.5)

except KeyboardInterrupt:
    GPIO.cleanup()
    print("Stopped")

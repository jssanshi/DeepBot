"""GPIO Controller for L298"""
import RPi.GPIO as GPIO
import time


# Left (BCM)
LEFT_IN3 = 27
LEFT_IN4 = 22
LEFT_ENB = 17

# Right (BCM)
RIGHT_ENA = 18
RIGHT_IN1 = 14
RIGHT_IN2 = 15

PINS = {"L": {"EN": LEFT_ENB, "IN1": LEFT_IN3, "IN2": LEFT_IN4},
        "R": {"EN": RIGHT_ENA, "IN1": RIGHT_IN1, "IN2": RIGHT_IN2}}

BALANCE_SPEED = 2.5
INIT_SPEED = 10

class MotorController(object):
    """GPIO Controller for L298."""

    def __init__(self):
        super(MotorController, self).__init__()

        GPIO.setmode(GPIO.BCM)
        for k in PINS.keys():
            GPIO.setup(PINS[k]["EN"],  GPIO.OUT)
            GPIO.setup(PINS[k]["IN1"], GPIO.OUT)
            GPIO.setup(PINS[k]["IN2"], GPIO.OUT)
        self.pwm = {"L": GPIO.PWM(PINS["L"]["EN"], 20),
                    "R": GPIO.PWM(PINS["R"]["EN"], 20)}

    def __del__(self):
        self.stop()
        self.cleanup()

    def stop(self):
        for k in PINS.keys():
            GPIO.output(PINS[k]["EN"],  GPIO.LOW)
            GPIO.output(PINS[k]["IN1"], GPIO.LOW)
            GPIO.output(PINS[k]["IN2"], GPIO.LOW)
        print("=====stop=====")

    def forcebreak(self):
        for k in PINS.keys():
            GPIO.output(PINS[k]["EN"],  GPIO.HIGH)
            GPIO.output(PINS[k]["IN1"], GPIO.LOW)
            GPIO.output(PINS[k]["IN2"], GPIO.LOW)

    def _forward(self, k, dc=INIT_SPEED):
        GPIO.output(PINS[k]["IN1"], GPIO.HIGH)
        GPIO.output(PINS[k]["IN2"], GPIO.LOW)
        self.pwm[k].start(dc)

    def _reverse(self, k, dc=INIT_SPEED):
        GPIO.output(PINS[k]["IN1"], GPIO.LOW)
        GPIO.output(PINS[k]["IN2"], GPIO.HIGH)
        self.pwm[k].start(dc)

    def forward(self, duration=None, dc=INIT_SPEED):
        self._forward("L", dc + BALANCE_SPEED)
        self._forward("R", dc)
        if duration:
            time.sleep(duration)

    def reverse(self, duration=None, dc=INIT_SPEED):
        self._reverse("L", dc + BALANCE_SPEED)
        self._reverse("R", dc)
        if duration:
            time.sleep(duration)

    def rotate(self, duration=None, dc=INIT_SPEED):
        self._reverse("L", dc + BALANCE_SPEED)
        self._forward("R", dc)
        if duration:
            time.sleep(duration)

    def turn_l(self, radius=0, duration=None):
        if radius > 0:
            self._forward("R", radius)
            self._forward("L", 0)
        else:
            self._reverse("L")
            self._forward("R")
        if duration:
            time.sleep(duration)

    def turn_r(self, radius=0, duration=None):
        if radius > 0:
            self._forward("R", 0)
            self._forward("L", radius)
        else:
            self._forward("L")
            self._reverse("R")
        if duration:
            time.sleep(duration)

    def cleanup(self):
        GPIO.cleanup()
        print("=====cleanup=====")


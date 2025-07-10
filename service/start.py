# 摄像机实时画面
from flask import Flask, render_template, Response
import cv2
from motor import MotorController

app = Flask(__name__)
mot = MotorController()
SPEED = 15

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('control.html')

@app.route("/ctrl/<state>")
def ctrl(state):
    if state == "t_up":
        mot.forward(dc=SPEED)
    elif state == "t_down":
        mot.reverse(dc=SPEED)
    elif state == "t_left":
        mot.turn_l(radius=SPEED)
    elif state == "t_right":
        mot.turn_r(radius=SPEED)
    elif state == 't_rotate':
        mot.rotate(dc=SPEED)
    elif state == 't_stop':
        mot.stop()
    return 'success'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port =8000, debug=True, threaded=True)
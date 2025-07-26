# 摄像机实时画面
from flask import Flask, render_template, Response
import cv2
from motor import MotorController
import time

app = Flask(__name__)
mot = MotorController()
SPEED = 21

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('control_video.html')

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

def gen_frames():
    camera = cv2.VideoCapture(0)
    # 可尝试设置采集格式为 MJPEG（更兼容）
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while True:
        success, frame = camera.read()
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        if not success:
            print("Failed to read frame")
            continue

        try:
            # 确保图像是 BGR 格式，不是灰度或压缩格式
            if len(frame.shape) == 2 or frame.shape[2] == 1:
                # 单通道图像 → 转 BGR（虽然不常见）
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 2:
                # 可能是 YUYV → 转换
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)

            # 缩小分辨率避免编码异常
            frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)

            # 编码为 JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except Exception as e:
            print(f"[ERROR] Encoding failed: {e}")

        time.sleep(0.1)  # 降帧率到约 10 fps

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port =8000, debug=True, threaded=True)
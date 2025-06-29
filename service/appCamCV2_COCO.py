# 摄像机实时目标检测
import threading
import time
import eventlet

import cv2
from flask import Flask, render_template, Response
from pycoral.adapters import common, detect
from pycoral.utils.edgetpu import make_interpreter

app = Flask(__name__)

# 配置
MODEL_PATH = 'model/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
LABEL_PATH = 'model/coco_labels.txt'
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MAX_WIDTH = 1920
MAX_HEIGHT = 1080
latest_detection_text = "无识别结果"

# 初始化 TPU 模型
interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()
interpreter_lock = threading.Lock()

# 加载标签
def load_labels(path):
    labels = {}
    with open(path, 'r') as f:
        for line in f:
            pair = line.strip().split(maxsplit=1)
            if len(pair) == 2:
                labels[int(pair[0])] = pair[1]
    return labels

labels = load_labels(LABEL_PATH)

# 初始化摄像头
camera = cv2.VideoCapture(0)
# 可尝试设置采集格式为 MJPEG（更兼容）
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not camera.isOpened():
    raise RuntimeError("无法打开摄像头。请检查连接。")

def process_frame(frame):
    if frame is None:
        print("空帧")
        return None

    # 灰度图转BGR三通道
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        print(f"非三通道图像，转换为BGR，shape={frame.shape}")
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # 固定大小，防止尺寸过大导致编码失败
    if frame.shape[1] > MAX_WIDTH or frame.shape[0] > MAX_HEIGHT:
        print(f"图像过大 {frame.shape[1]}x{frame.shape[0]}，缩放中")
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    resized = cv2.resize(frame, common.input_size(interpreter))
    rgb_input = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    common.set_input(interpreter, rgb_input)

    with interpreter_lock:
        start = time.time()
        interpreter.invoke()
        inference_time = time.time() - start
        objs = detect.get_objects(interpreter, score_threshold=0.5)

    h, w, _ = frame.shape
    scale_x = w / common.input_size(interpreter)[0]
    scale_y = h / common.input_size(interpreter)[1]

    results = []

    for obj in objs:
        bbox = obj.bbox
        x0, y0 = int(bbox.xmin * scale_x), int(bbox.ymin * scale_y)
        x1, y1 = int(bbox.xmax * scale_x), int(bbox.ymax * scale_y)
        label = labels.get(obj.id, f"ID {obj.id}")
        results.append(f"{label} ({obj.score:.2f})")
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(frame, f'{label} {obj.score:.2f}', (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 设置识别结果文本
    global latest_detection_text
    if results:
        latest_detection_text = "，".join(results)
    else:
        latest_detection_text = "未识别到物体"

    cv2.putText(frame, f"Inference: {inference_time*1000:.1f} ms",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return frame

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success or frame is None:
            print("读取摄像头帧失败")
            continue

        # 只有3维（H,W,3）才旋转，否则跳过
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        else:
            print(f"跳过旋转，frame shape异常: {frame.shape}")

        # 如果是灰度图，转换成BGR三通道
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # 防止尺寸过大
        if frame.shape[1] > MAX_WIDTH or frame.shape[0] > MAX_HEIGHT:
            print(f"图像过大 {frame.shape[1]}x{frame.shape[0]}，缩放中")
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        processed = process_frame(frame)
        if processed is None:
            continue

        ret, buffer = cv2.imencode('.jpg', processed)
        if not ret:
            print("编码帧失败")
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        eventlet.sleep(0.01)

def get_latest_detection_result():
    global latest_detection_text
    return latest_detection_text

@app.route('/')
def index():
    return render_template('camera_tpu.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/model_output')
def model_output():
    return get_latest_detection_result()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)

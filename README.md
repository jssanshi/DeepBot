# DeepBot
## 机器人运行 start.py
1. python3.9环境安装依赖:

    ```
    pip install -r requirements.txt
    ```
2. 将service目录拷贝到树莓派目录下
3. 切换到service目录下，启动机器人服务:

    ```
    python start.py
    ```
4.  浏览器访问网页，点击对应按钮操作:

    ```
    http://localhost:8000
    ```
## 实时展示摄像机画面 appCamCV2.py 
1.  python3.9运行脚本:

    ```
    python appCamCV2.py
    ```
2.  浏览器访问网页，查看实时画面:

    ```
    http://localhost:8000
    ```
## 实时展示摄像机目标检测画面 appCamCV2_COCO.py 
1. 修改tflite文件路径，coco_labels.txt文件路径:

    ```
    MODEL_PATH = 'model/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    LABEL_PATH = 'model/coco_labels.txt'
    ```
2. 安装依赖，pycoral安装过程需参考[Coral USB加速器安装调试](https://coral.ai/docs/accelerator/get-started/#3-run-a-model-on-the-edge-tpu).
3. python3.9运行脚本:

    ```
    python appCamCV2_COCO.py
    ```
4. 浏览器访问网页，查看实时目标检测画面:

    ```
    http://localhost:8000
    ```
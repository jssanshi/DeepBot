import tensorflow as tf
import time
import numpy as np
from PIL import Image

#Loading the saved_model
with open("../data/model_result/road_sign_labels.txt", 'r') as f:
    pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
    labels = dict((int(k), {"id": int(k), "name": v}) for k, v in pairs)

PATH_TO_SAVED_MODEL="../data/model_result/saved_model"
print('Loading model...', end='')
# Load saved model and build the detection function
detect_fn=tf.saved_model.load(PATH_TO_SAVED_MODEL)
print('Done!')

start_ms = time.time()

image_np = np.array(Image.open("../data/images/train/images/camera_001.jpg"))
# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
input_tensor = tf.convert_to_tensor(image_np)
# The model expects a batch of images, so add an axis with `tf.newaxis`.
input_tensor = input_tensor[tf.newaxis, ...]

detections = detect_fn(input_tensor)
num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}

my_classes = detections['detection_classes']
my_scores = detections['detection_scores']
min_score = 0.9
print([labels[value]['name']
       for index,value in enumerate(my_classes)
       if my_scores[index] > min_score
       ])

end_tf_ms = time.time()
elapsed_tf_ms = end_tf_ms - start_ms
print(elapsed_tf_ms)
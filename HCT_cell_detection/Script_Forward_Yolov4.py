# By Rohollah Moosavi Tayebi, email: rohollah.moosavi@uwaterloo.ca/moosavi.tayebi@gmail.com

# This script gets a tile and returns detected and classified cells inside
# and saves it.

import cv2
import numpy as np
import tensorflow as tf
import time
from yolov4.tf import YOLOv4
patch_number = "17142"
yolo = YOLOv4()
yolo.classes = "./model/obj.names"
yolo.make_model()
yolo.load_weights("./model/yolo-obj_best.weights", weights_type="yolo")

print(yolo.classes)

original_image = cv2.imread("/home/weber50432/AML_image_processing/HCT_region_select/output/A3/ROI/A3_{}.png".format(patch_number))
resized_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
resized_image = yolo.resize_image(resized_image)
resized_image = resized_image / 255
input_data = resized_image[np.newaxis, ...].astype(np.float32)

start_time = time.time()
candidates = yolo.model.predict(input_data)
_candidates = []
for candidate in candidates:
    batch_size = candidate.shape[0]
    grid_size = candidate.shape[1]
    _candidates.append(
        tf.reshape(
            candidate, shape=(1, grid_size * grid_size * 3, -1)
        )
    )

candidates = np.concatenate(_candidates, axis=1)
pred_bboxes = yolo.candidates_to_pred_bboxes(candidates[0])
pred_bboxes = yolo.fit_pred_bboxes_to_original(
    pred_bboxes, original_image.shape
)
end_time = time.time()
elapsed_time = end_time - start_time
print('cells were detected from the ROI A3_{}, time:{:02d}m{:02d}s'.format(patch_number,int(elapsed_time // 60), int(elapsed_time % 60)))
result = yolo.draw_bboxes(original_image, pred_bboxes)

cv2.imwrite(f"./output/A3_{patch_number}.png", result)
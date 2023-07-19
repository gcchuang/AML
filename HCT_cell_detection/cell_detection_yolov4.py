# This script gets a tile and returns detected and classified cells inside and saves it. script is for v2.x.x. of YOLOv4
import cv2
import sys
import os
import numpy as np
import tensorflow as tf
import pickle
import time
from yolov4.tf import YOLOv4
# function to crop the cellsq
def crop_cells(image_name,image,bboxes,classes,out_dir = "./",cell_size = 64,cell_propotion = 0.8):
    if os.path.exists(out_dir) == False:
        os.makedirs(out_dir)
    num_boxes = bboxes
    image_h, image_w, color = image.shape
    for i in range(num_boxes.shape[0]):
        start_x = int(num_boxes[i][0] * image_w)
        start_y = int(num_boxes[i][1] * image_h)
        cell_w = int(num_boxes[i][2] * image_w)
        cell_h = int(num_boxes[i][3] * image_h)
        # check the cell size, if the cell is too small or too big, skip it
        if cell_w < (cell_size*cell_propotion) or cell_h < (cell_size*cell_propotion):
            # print("cell too small",cell_w,cell_h)
            continue
        elif cell_w > (cell_size/cell_propotion) or cell_h > (cell_size/cell_propotion):
            # print("cell too big",cell_w,cell_h)
            continue
        elif num_boxes[i][5]<0.5:
            continue
        #check the cell type by column 4, if the directory does not exist, create one
        if os.path.exists(out_dir+f"{classes[int(num_boxes[i][4])]}") == False:
            os.makedirs(out_dir+f"{classes[int(num_boxes[i][4])]}")
        # crop the cell depends on the larger edge
        if cell_w > cell_h:
            sub_image = image[start_y-cell_w//2:start_y+cell_w//2 , start_x-cell_w//2:start_x+cell_w//2]
        else:
            sub_image = image[start_y-cell_h//2:start_y+cell_h//2 , start_x-cell_h//2:start_x+cell_h//2]
        if sub_image.shape[0] == 0 or sub_image.shape[1] == 0 or sub_image.shape[2] == 0:
            continue
        # resize the cell to 64*64
        # print(sub_image.shape)
        sub_image = cv2.resize(sub_image,(cell_size,cell_size))
        # save the cell image
        cv2.imwrite(f"{out_dir+classes[int(num_boxes[i][4])]}/{image_name}_{i+1}.png", sub_image)

yolo = YOLOv4()
yolo.classes = "./model/obj.names"
classes_file = "./model/obj.names"
with open(classes_file, 'r') as f:
    classes = f.read().splitlines()
yolo.make_model()
yolo.load_weights("./model/yolo-obj_best.weights", weights_type="yolo")
# select the cell from the single patch
slide_num = "A26"
patch_number = "36601"
original_image = cv2.imread("/home/exon_storage1/aml_slide/ROI_level0_pixel512/{}/{}_{}.png".format(slide_num,slide_num,patch_number))
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
print(pred_bboxes)
result = yolo.draw_bboxes(original_image, pred_bboxes)
# crop_cells(f"{slide_num}_{patch_number}",original_image, pred_bboxes,f"./output/{slide_num}_{patch_number}/")
cv2.imwrite(f"./output/{slide_num}_{patch_number}.png", result)
# counting the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print('cells were detected from the ROI {}_{}, time:{:02d}m{:02d}s'.format(slide_num,patch_number,int(elapsed_time // 60), int(elapsed_time % 60)))
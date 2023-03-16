#this script is used to normalize the patches and save them in a new folder, with a blurry filter
import pandas as pd
import numpy as np
import os
import re
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
sys.path.insert(1, '/home/weber50432/AML_image_processing/HEnorm_python/')
from normalizeStaining import normalizeStaining , is_blurry
directory_path = '/home/exon_storage1/aml_slide/patches/'
for file in os.listdir(directory_path):
    if  os.path.exists(directory_path+file+"/rightside_patch"):
        patches_path = directory_path+file+"/rightside_patch/"
        if os.path.exists(directory_path+file+"/rightside_patch_norm"):
            print(file + " rightside_patch_norm is already created !")
            continue
        else:
            os.mkdir(directory_path+file+"/rightside_patch_norm/")
        if os.path.exists(directory_path+file+"/rightside_patch_blurry"):
            print(file + " rightside_patch_blurry is already created !")
        else:
            os.mkdir(directory_path+file+"/rightside_patch_blurry/")
        patches_path_blurry = directory_path+file+"/rightside_patch_blurry/"
        patches_path_norm = directory_path+file+"/rightside_patch_norm/"
        for img in os.listdir(patches_path):
            if img.endswith(".png"):
                img_num = str(img)
                img_path = patches_path+img
                img = np.array(Image.open(img_path))
                Inorm, H, E = normalizeStaining(img = img,
                      saveFile = None,
                      Io = 240,
                      alpha = 1,
                      beta = 0.15)
                if is_blurry(Inorm):
                    Image.fromarray(Inorm).save(patches_path_blurry+img_num)
                    # os.remove(img_path)
                    print(img_num+" is blurry !")
                else:
                    Image.fromarray(Inorm).save(patches_path_norm+img_num)
                    print(img_num+" is saved !")
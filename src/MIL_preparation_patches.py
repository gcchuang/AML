# %%
import pandas as pd
import numpy as np
import os
import time
import torch
from sklearn.model_selection import train_test_split
import utility_functions as uf
start_time = time.time()
# %%
patches_size = 512
target_gene = "NPM1"
left_proportion = 0.6
shrink_proportion = 0.15
patch_num = 500
level = 0
grid = []
slide_path = "/home/weber50432/AML_image_processing/HCT_region_select/output/"
info_file_path = "/home/weber50432/AML_image_processing/code_use_csv/changeSlideName.csv"
grid_val = []
# slide_list = uf.get_slides_list_number(f"{slide_path}otsu_1.0/")
slide_list_train = [3, 9, 12, 13, 22, 26, 27, 28, 29, 60, 99, 102, 103, 104, 105, 106, 108, 109, 113, 121, 123, 129, 130, 131, 133, 134, 135, 137, 138, 140, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 156, 158, 159, 160, 161, 162, 163, 165, 166, 167, 168, 169, 171, 172, 173, 174, 176, 177, 178, 180, 197, 199, 200, 201, 203, 204, 205, 206, 207, 208, 209, 212, 217, 219, 220, 227, 230, 231, 232, 233, 234, 235, 238, 239, 241, 242, 244, 265, 273, 274, 280, 299, 314, 317, 319, 320, 322, 323, 330, 331, 332, 333, 334, 335, 336, 337, 338, 341, 342, 343]
slide_list_val = [ 344, 345, 347, 348, 349, 350, 352, 353, 355, 357, 359, 360, 361, 363, 364, 365, 366, 367, 369, 370, 371, 372, 373, 374, 377, 378, 380, 381, 382, 383, 386, 387, 390, 392, 393, 395, 1001]
slide_list_test = [25,59,124,136,143,224,257,292,308,311,326,358,362,368,384,385,396,403,404,405,406,408,409,410,412,413,415,420,422,423,424,425,426,427,428,429,430,431,437,438,439,440]
target_gene_rename = target_gene+"_patch_test"
output_path ="/home/weber50432/AML_image_processing/lib/{}".format(target_gene_rename)
# check the output path is exist or not
if not os.path.exists(output_path):
    os.makedirs(output_path)
# save the data
# train_output = {
#       "slides": uf.make_paths_list("",slide_list_train),
#       "grid": uf.get_patches_grid(slide_path,slide_list_train,patch_num),
#       "targets": uf.get_targets_list(target_gene,slide_list_train,info_file_path),
#       "mult": patches_size/224,
#       "level": level,
#   }
# torch.save(train_output, "{}/{}_train_data.pt".format(output_path,target_gene_rename))
# print("training data is done")
# val_output = {
#       "slides": uf.make_paths_list("",slide_list_val),
#       "grid": uf.get_patches_grid(slide_path,slide_list_val,patch_num),
#       "targets": uf.get_targets_list(target_gene,slide_list_val,info_file_path),
#       "mult": patches_size/224,
#       "level": level,
#   }
# torch.save(val_output, "{}/{}_val_data.pt".format(output_path,target_gene_rename))
# print("validation data is done")
test_output = {
      "slides": uf.make_paths_list("",slide_list_test),
      "grid": uf.get_patches_grid(slide_path,slide_list_test,patch_num),
      "targets": uf.get_targets_list(target_gene,slide_list_test,info_file_path),
      "mult": patches_size/224,
      "level": level,
  }
torch.save(test_output, "{}/{}_test_data.pt".format(output_path,target_gene_rename))
print("test data is done")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"程式執行時間：{elapsed_time:.2f} 秒")


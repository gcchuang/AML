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
slide_list = [3, 9, 12, 13,26, 27, 28, 29,60, 99, 102,104, 219,1001]
target = uf.get_targets_list(target_gene,slide_list,info_file_path)
target_gene_rename = target_gene+"_patch_test"
output_path ="/home/weber50432/AML_image_processing/lib/{}".format(target_gene_rename)
# check the output path is exist or not
if not os.path.exists(output_path):
    os.makedirs(output_path)
# save the data
train_output = {
      "slides": uf.make_paths_list("",slide_list),
      "grid": uf.get_patches_grid(slide_path,slide_list,patch_num),
      "targets": target,
      "mult": patches_size/224,
      "level": level,
  }
torch.save(train_output, "{}/{}_train_data.pt".format(output_path,target_gene_rename))
print("train_output is done")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"程式執行時間：{elapsed_time:.2f} 秒")


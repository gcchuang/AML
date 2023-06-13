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
# left_proportion = 0.6
# shrink_proportion = 0.15
patch_num = 2000
level = 0
slide_path = "/home/exon_storage1/aml_slide/"
info_file_path = "/home/weber50432/AML_image_processing/code_use_csv/changeSlideName.csv"
slide_list = uf.get_slides_list_number(f"{slide_path}ROI_level0_pixel512/")
target = uf.get_targets_list(target_gene,slide_list,info_file_path)

# %%
X = np.array(slide_list)
y = np.array(target)
# 將數據集按比例 6:2:2 分為訓練集、驗證集和測試集
# 找到標籤為1的樣本索引
positive_indices = np.where(y == 1)[0]

# 使用 train_test_split 函数分割樣本
# 首先將標籤為1的樣本分成訓練集、驗證集和測試集
train_pos, val_pos_test = train_test_split(positive_indices, test_size=0.4)
val_pos, test_pos = train_test_split(val_pos_test, test_size=0.5)

# 接下來將標籤為0的樣本分成訓練集、驗證集和測試集
train_neg, val_neg_test, y_train, y_val_test = train_test_split(np.where(y == 0)[0], y[np.where(y == 0)[0]], test_size=0.4)
val_neg, test_neg, y_val, y_test = train_test_split(val_neg_test, y_val_test, test_size=0.5)

# 將訓練集、驗證集和測試集的索引合併起來
train_indices = sorted(np.concatenate((train_pos, train_neg)))
val_indices = sorted(np.concatenate((val_pos, val_neg)))
test_indices = sorted(np.concatenate((test_pos, test_neg)))

# 根據索引提取對應的數據和標籤
X_train = X[train_indices].tolist()
X_val = X[val_indices].tolist()
X_test = X[test_indices].tolist()
y_train = y[train_indices].tolist()
y_val = y[val_indices].tolist()
y_test = y[test_indices].tolist()

# 計算各個集合的樣本數量
print(f"訓練集樣本數量：{len(X_train)}")
print(f"positive target：{y_train.count(1)}")
print(f"驗證集樣本數量：{len(X_val)}")
print(f"positive target：{y_val.count(1)}")
print(f"測試集樣本樣量：{len(X_test)}")
print(f"positive target：{y_test.count(1)}")

# %%
target_gene_rename = target_gene.split(" ")[0]+"_patch_20000"
output_path ="/home/weber50432/AML_image_processing/lib/{}".format(target_gene_rename)
# check the output path is exist or not
if not os.path.exists(output_path):
    os.makedirs(output_path)
# save the data
train_output = {
      "slides": uf.make_paths_list("/staging/biology/b08611005/ROI_level0_pixel512/",X_train),
      "grid": uf.get_patches_grid(slide_path+"ROI_level0_pixel512/",X_train,patch_num),
      "targets": y_train,
      "mult": patches_size/224,
      "level": level,
  }
torch.save(train_output, "{}/{}_train_data.pt".format(output_path,target_gene_rename))
print("train_output is done")
val_output = {
      "slides": uf.make_paths_list("/staging/biology/b08611005/ROI_level0_pixel512/",X_val),
      "grid": uf.get_patches_grid(slide_path+"ROI_level0_pixel512/",X_val,patch_num),
      "targets": y_val,
      "mult": patches_size/224,
      "level": level,
  }
torch.save(val_output, "{}/{}_val_data.pt".format(output_path,target_gene_rename))
print("val_output is done")
test_output = {
      "slides": uf.make_paths_list("/staging/biology/b08611005/ROI_level0_pixel512/".format(slide_path),X_test),
      "grid": uf.get_patches_grid(slide_path+"ROI_level0_pixel512/",X_test,patch_num),
      "targets": y_test,
      "mult": patches_size/224,
      "level": level,
  }
torch.save(test_output, "{}/{}_test_data.pt".format(output_path,target_gene_rename))
print("test_output is done")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"程式執行時間：{elapsed_time:.2f} 秒")


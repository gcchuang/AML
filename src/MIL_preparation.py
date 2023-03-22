import pandas as pd
import numpy as np
import os
import torch
import utility_functions as uf
patches_size = 256
slide_path_otsu_1 = "/home/exon_storage1/aml_slide/otsu_1.0/"
slide_path_otsu_2 = "/home/exon_storage1/aml_slide/otsu_0.9/"
slide_list = uf.get_slides_list_number(slide_path_otsu_1)
slide_list2 = uf.get_slides_list_number(slide_path_otsu_2)
grid = []
if slide_list == slide_list2:
  slide_list_out = uf.make_slides_list("/home/exon_storage1/aml_slide/renameByUPN/",slide_list)
  target = uf.get_targets_list("DNMT3A",slide_path_otsu_1,"/home/weber50432/AML_image_processing/code_use_csv/changeSlideName.csv")
  for slide in os.listdir(slide_path_otsu_1):
    path1 = '/home/exon_storage1/aml_slide/otsu_0.9/{}/tile_selection.tsv'.format(slide)
    path2 = '/home/exon_storage1/aml_slide/otsu_1.0/{}/tile_selection.tsv'.format(slide)
    list1, num1 = uf.get_coordinates(path1,patches_size,0.6,0.2)
    list2, num2 = uf.get_coordinates(path2,patches_size,0.6,0.2)
    set1 = set(list1)
    set2 = set(list2)
    difference = set1.difference(set2)
    result = list(difference)
    grid.append(result)
    print("slide {} done, patch_number are {}.".format(slide,len(result)))
  output = {
      "slides": slide_list_out,
      "grid": grid,
      "targets": target,
      "mult": patches_size/224,
      "level": 0,
  }
  torch.save(output, "./data.pt")
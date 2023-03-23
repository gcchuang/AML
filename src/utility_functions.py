import pandas as pd
import numpy as np
import torch
import os
import sys
from PIL import Image
# get the absolute path of the slides in the slide_list
def make_paths_list(slide_path ,slide_list):
  slide_list.sort()
  slide_list = [slide_path + "A" +str(x) +".ndpi" for x in slide_list]
  return slide_list
# get the numbers of slides in a particular path
def get_slides_list_number(slide_path):
  slide_list = os.listdir(slide_path)
  slide_list = [int("".join(filter(str.isdigit, i))) for i in slide_list]
  slide_list.sort()
  return slide_list
#get target gene targets list ( 0 : benign slide, 1: tumor slide )
def get_targets_list(gene,slide_list,info_file_path):
  df = pd.read_csv(info_file_path, sep=",", encoding="utf-8")
  result = df[df["UPN, c4lab"].isin(slide_list)]
  targets_list = result[gene].astype(int).tolist()
  return targets_list
# this is a function to get the coordinates of the patches in the WSIs, according to the tile_selection.tsv file
def get_coordinates(tile_selection_path,patches_size,left_proportion,top_proportion):
    df = pd.read_table(tile_selection_path)
    # select the patches with Keep = 1 and in the lower right side of the WSI based on the proportion
    df = df.loc[(df["Keep"] == 1) & (df["Column"]>=int(max(df["Column"])*left_proportion))& (df["Row"]>=int(max(df["Row"])*top_proportion))]
    # group the rows by index to form a list of tuples
    tuples = [(x, y) for x, y in zip(df['Row']*patches_size, df['Column']*patches_size)]
    patches_num = len(tuples)
    return tuples , patches_num
def subtract_two_coordinates(list1,list2):
  list1 = list(set(list1) - set(list2))
  return list1
def get_grid(slide_path,slide_list,patches_size,left_proportion,top_proportion):
  grid = []
  for slide in slide_list:
    path1 = '{}otsu_0.9/A{}/tile_selection.tsv'.format(slide_path,slide)
    path2 = '{}otsu_1.0/A{}/tile_selection.tsv'.format(slide_path,slide)
    list1, num1 = get_coordinates(path1,patches_size,left_proportion,top_proportion)
    list2, num2 = get_coordinates(path2,patches_size,left_proportion,top_proportion)
    result = subtract_two_coordinates(list1,list2)
    grid.append(result)
    if num1 - num2 == len(result):
        print("slide {} done, patch_number are {}.".format(slide,len(result)))
    else:
        print("slide {} done, patch_number are {}, but the difference is {}.".format(slide,len(result),num1-num2))
  return grid

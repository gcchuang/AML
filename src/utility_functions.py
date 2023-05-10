import pandas as pd
import numpy as np
import os
import random
import xlrd
# get the absolute path of the slides in the slide_list
def make_slides_list(slide_path ,slide_list):
  slide_list.sort()
  slide_list = [slide_path + "A" +str(x) +".ndpi" for x in slide_list]
  return slide_list
# get the absolute path of the patches in the slide_list
def make_paths_list(slide_path ,slide_list):
  slide_list.sort()
  slide_list = [slide_path + "A" +str(x) for x in slide_list]
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
def get_coordinates(tile_selection_path,patches_size,left_proportion,shrink_proportion):
    df = pd.read_table(tile_selection_path)
    row_max = (max(df["Row"]))
    col_max = (max(df["Column"]))
    # select the patches with Keep = 1
    df = df.loc[(df["Keep"] == 1)]
    # select the patches in the right side of the slide
    df = df.loc[(df["Column"]>=int(col_max*left_proportion))]
    df = df.loc[(df["Column"]<=int(col_max*(1-(1-left_proportion)*shrink_proportion)))]
    # shrink the sleceted area
    df = df.loc[(df["Row"]>=int(row_max*shrink_proportion))& (df["Row"]<=int(row_max*(1-shrink_proportion)))]
    # group the rows by index to form a list of tuples
    tuples = [(x, y) for x, y in zip(df['Column']*patches_size, df['Row']*patches_size)]
    patches_num = len(tuples)
    return tuples , patches_num
def subtract_two_coordinates(list1,list2):
  list1 = list(set(list1) - set(list2))
  return list1
def get_grid(slide_path,slide_list,patches_size,left_proportion,shrink_proportion,patch_num):
  grid = []
  for slide in slide_list:
    path1 = '{}otsu_0.9/A{}/tile_selection.tsv'.format(slide_path,slide)
    path2 = '{}otsu_1.0/A{}/tile_selection.tsv'.format(slide_path,slide)
    list1, num1 = get_coordinates(path1,patches_size,left_proportion,shrink_proportion)
    list2, num2 = get_coordinates(path2,patches_size,left_proportion,shrink_proportion)
    result = subtract_two_coordinates(list1,list2)
    if patch_num < len(result):
        result = random.sample(result,patch_num)
        print("slide {} done, slect {} patches from {} total patches.".format(slide,len(result),num1-num2))
    else:
        print("slide {} done, patch_number are {}.".format(slide,len(result)))
    grid.append(result)
  return grid
def get_patches_grid_excel(slide_path,slide_list,patch_num):
  grid = []
  for slide in slide_list:
    slide_name = "A" + str(slide)
    print("slide {} is processing...".format(slide_name))
    if slide_name in os.listdir(slide_path):
      excel_path = slide_path + slide_name + "/porduction_result.xls"
      if os.path.exists(excel_path):
        wb = xlrd.open_workbook(excel_path)
        sheet = wb.sheet_by_index(0)
        patch_list = sheet.col_values(0)
        patch_list.pop(0)
        if patch_num < len(patch_list):
          patch_list = random.sample(patch_list,patch_num)
        grid.append(patch_list)
      else:
        print("slide {} do not have excel file.".format(slide_name))
  return grid
def get_patches_grid(slide_path,slide_list,patch_num):
  grid = []
  for slide in slide_list:
    slide_name = "A" + str(slide)
    print("slide {} is processing...".format(slide_name))
    if slide_name in os.listdir(slide_path):
      patch_list = os.listdir(slide_path + slide_name)
      if patch_num < len(patch_list):
        patch_list = random.sample(patch_list,patch_num)
      grid.append(patch_list)
    else:
      print("slide {} do not exist.".format(slide_name))
  return grid
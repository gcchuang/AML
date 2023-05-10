import os, shutil
import xlrd
import time
import sys
# this script is used to select the ROI patches from the whole slide images, and save them in a new folder named ROI
def copy_files(output_path,exel_source,slidenum):
    os.mkdir("{}{}".format(output_path,slidenum)) # create a new folder for each slide
    dest_Fold = "{}{}/".format(output_path,slidenum)
    wb = xlrd.open_workbook(exel_source)
    sheet = wb.sheet_by_index(0)
    patch_list = sheet.col_values(0)
    patch_list.pop(0)
    for patch in patch_list:
      shutil.copy(patch, dest_Fold)
    return len(patch_list)
    
if __name__== "__main__":
  log_lib = "/home/weber50432/AML_image_processing/log/"
  sys.stdout = open("{}ROI_selecting_output.log".format(log_lib), 'w', encoding='utf-8')
  sys.stderr = open("{}ROI_selecting_error.log".format(log_lib), 'w', encoding='utf-8')
  excel_file_path = '/home/weber50432/AML_image_processing/HCT_region_select/output/'
  output_path = '/home/exon_storage1/aml_slide/ROI_level0_pixel512/'
  for slidenum in os.listdir(excel_file_path):
    if os.path.exists("{}{}".format(output_path,slidenum)):
      print('{} is already processed !'.format(slidenum))
      continue  
    exel_source = "{}{}/porduction_result.xls".format(excel_file_path,slidenum)
    if not os.path.exists(exel_source):
      print('{} does not have the production_result.xls file!'.format(slidenum))
      continue
    start_time = time.time()
    print('{} is processing...'.format(slidenum))
    patches_num = copy_files(output_path,exel_source,slidenum)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('{} patches were copied to the ROI directory, time:{:02d}m{:02d}s'.format(patches_num,int(elapsed_time // 60), int(elapsed_time % 60)))
import pandas as pd
import os
# this script is used to select the right side patches from the whole slide images, and save them in a new folder named rightside_patch
directory_path = '/home/exon_storage1/aml_slide/patches/'
for file in os.listdir(directory_path):
  if os.path.isdir(directory_path+file):
    if os.path.exists(directory_path+file+'/rightside_patch/'):
      print(file+' is already processed !')
      continue  
    else:
      print(file+' is being processed !')
      df = pd.read_table(directory_path+file+'/tile_selection.tsv')
      df = df.loc[(df["Keep"] == 1) & (df["Column"]>=int(max(df["Column"])/2))]
      count = 0
      patch_path = directory_path+file+'/'+file+'_tiles/'
      os.mkdir(directory_path+file+'/rightside_patch/')
      output_path = directory_path+file+'/rightside_patch/'
      tile_list = df.Tile.values.tolist()
      for img in os.listdir(patch_path):
        if (img[:img.index(".")])in tile_list:
          count += 1
          old_name = patch_path+img
          new_name = output_path+img
          os.rename(old_name,new_name)
      if count != len(tile_list):
        print("Error: Not all tiles are selected:"+str(count))
      else:
        print("All tiles are selected:"+ str(count))



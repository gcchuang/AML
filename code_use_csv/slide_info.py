import re
import os
import pandas as pd
def get_slides_list(slide_path):
  slide_list = os.listdir(slide_path)
  slide_list = [int("".join(filter(str.isdigit, i))) for i in slide_list]
  slide_list.sort()
  # print(slide_list[:5])
  # print(len(slide_list))
  return slide_list
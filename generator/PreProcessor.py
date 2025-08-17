# Copyright 2025 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 2025/08/05 Preprocessor.py

import os
import glob
import shutil
import numpy as np
import cv2
import csv
import traceback

class PreProcessor:

  def __init__(self, size=512):
    self.size   = size
    self.RESIZE = (size, size)
    self.bgr_map = {"BE":(0,255,0), "suspicious":(255,0,0), "HGD":(255,255,0), "cancer":(0,0,255), "polyp":(0,255,255)}


  def gamma_correction(self, img, gamma):
    table = (np.arange(256) / 255) ** gamma * 255
    table = np.clip(table, 0, 255).astype(np.uint8)
    return cv2.LUT(img, table)

  
  def sharpen(self, img, k=3):
    kernel =  np.array([
      [-k / 9, -k / 9, -k / 9],
      [-k / 9, 1 + 8 * k / 9, -k / 9],
      [-k / 9, -k / 9, -k / 9]
     ], np.float32)

    img = cv2.filter2D(img, ddepth=-1, kernel=kernel).astype("uint8")
    return img

  def crop_to_square(self, image):
    h, w, _ = image.shape
    min = h
    if min<w:
      min = w
    square = image[0:min, 0:min]
    #resize
    square = cv2.resize(square, self.RESIZE)
    return square
  

  def colorize_mask(self, mask, color=(255, 255, 255), gray=0):
    h, w = mask.shape[:2]
    print("mask shape {}".format(mask.shape))
    rgb_mask = np.zeros((h, w, 3), np.uint8)
    for x in range(w):
      for y in range(h):
        p = mask[y,x]
        if p>0:
          rgb_mask[y, x] = color

    return rgb_mask   
  
  def preprocess(self, images_dir, 
                 masks_dir, 
                 output_images_dir,
                 output_masks_dir):
  

    image_files = glob.glob(images_dir + "/*.jpg")
    #mask_files   = glob.glob(masks_dir + "/*.tif")
    image_files = sorted(image_files)
    #mask_files  = sorted(mask_files)
    num_image_files = len(image_files)
    #num_mask_files  = len(mask_files)

    print("--- num_image files {} ".format(num_image_files))

    #if num_image_files != num_mask_files:
    #  raise Exception("Unmatched num_image_files and num_mask_file")
    index = 1000
    
    for i in range(num_image_files):
      image_file = image_files[i]
      image      = cv2.imread(image_file)
      h, w, _    = image.shape
      image      = self.crop_to_square(image)
      output_name = str(index + i) + ".png"
      output_path = os.path.join(output_images_dir, output_name)
      cv2.imwrite(output_path, image)
      print("Saved {}".format(output_path))

      basename = os.path.basename(image_file)
      name     = basename.split(".")[0]
      mask_files = glob.glob(masks_dir + "/" + name + "*.tif")
      print("{}  {}".format(image_file,  mask_files))

      merged_mask = np.zeros((h, w, 3), dtype=np.uint8)
      for mask_file in mask_files:
        cname = os.path.basename(mask_file).split(".")[0]
        category = cname.split("_")[2]
        print("catgory {}".format(category))
        mask  = cv2.imread(mask_file)
        mask  = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        color = self.bgr_map[category]
        print("catgory {} color {}".format(category, color))

        mask  = self.colorize_mask(mask, color=color, gray=255)
      
        merged_mask += mask
      merged_mask =  self.crop_to_square(merged_mask)

      output_mask = os.path.join(output_masks_dir, output_name)
      cv2.imwrite(output_mask, merged_mask)
      print("Saved {}".format(output_mask))

      """
      mask = cv2.imread(mask_file)
      mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
      mask = cv2.resize(mask, self.RESIZE)
      basename = os.path.basename(mask_file)

      name     = basename.split(".") [0]
      name     = name.replace("mask", "Image")
      try:
        category = self.data[name]
        print("category {}".format(category)) 
        mask_color = self.bgr_map[category]

        mask  = self.colorize_mask(mask, color=mask_color, gray=255)

        filename = str(index + i) + ".png"
        output_mask_file = os.path.join(output_masks_dir, filename)
        cv2.imwrite(output_mask_file, mask)
        print("=== Saved {}".format(output_mask_file))
    
        image_file = image_files[i]
        img = cv2.imread(image_file)
        img = cv2.resize(img, self.RESIZE)
        
        output_image_file = os.path.join(output_images_dir, filename)
        img = self.sharpen(img)
        cv2.imwrite(output_image_file, img)
        print("=== Saved {}".format(output_image_file))
      
      except:
        traceback.print_exc()
        continue
      """

if __name__== "__main__":
  try:
    
    output_dir = "./EDD2020-PNG-master"
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    os.makedirs(output_dir)    

    output_images_dir = output_dir + "/images/"
    output_masks_dir = output_dir + "/masks/"

    os.makedirs(output_images_dir)
    os.makedirs(output_masks_dir)

    preprocessor = PreProcessor()

    images_dir = "./EDD2020/originalImages/"
    masks_dir  = "./EDD2020/masks/"
    preprocessor.preprocess(images_dir,
                            masks_dir,
                            output_images_dir,
                            output_masks_dir)
  except:
    traceback.print_exc()
    

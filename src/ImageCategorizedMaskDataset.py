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
# 2025/06/03
# 2025/06/07  Added an online rgb to categorized_mask function
#   def rgb_to_categorized_mask(self, rgb_mask_file):
#
# ImageCategorizedMaskDataset.py

import os
import sys
import glob
import shutil
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical

from ConfigParser import ConfigParser
from ImageMaskDataset import ImageMaskDataset
import tqdm

class ImageCategorizedMaskDataset(ImageMaskDataset):

  def __init__(self, config_file, verbose=False):
    super().__init__(config_file, verbose)
    #  rgb_map dict  { key1:value1, key2:value2,...}
    sample_rgb_map = { (0, 0, 0): 0, (0, 255, 0):1, (255,0,0):2, (0, 0, 255):3 }
    self.rgb_map   = self.config.get(ConfigParser.MASK, "rgb_map", dvalue=sample_rgb_map)
    print("--- rgb_map {}".format(self.rgb_map))
    self.rgb_color = []
    for item in self.rgb_map:
      self.rgb_color += [list(item)]  

    # rgb_color = [[0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255]]
    print("rgb_color {}".format(self.rgb_color))
  
    #self.num_classes = len(self.rgb_map)
    self.rgb_array   = np.array(self.rgb_color)
    self.rgb_array   = self.rgb_array.reshape((-1, 1, 1, 3))
    # Create a flattened palette
    self.palette = []
    for item in self.rgb_map:
      self.palette += list(item)
    print("--- palette {}".format(self.palette))

  def create(self, images_dir, masks_dir ):
    print("ImageCategorizedMaskDataset images_dir {} masks_dir {}".format(images_dir, masks_dir))

    image_files  = glob.glob(images_dir + "/*.jpg")
    image_files += glob.glob(images_dir + "/*.png")
    image_files += glob.glob(images_dir + "/*.tif")

    #print(image_files)
    num_images = len(image_files)

    mask_files = glob.glob(masks_dir  + "/*" + self.mask_file_format)

    num_masks  = len(mask_files)
    print("--- num_image_files: {}  num_mask_files:{}".format(num_images, num_masks))  
    if num_images != num_masks:
       error = "Unmatched the number of images and masks files."
       raise Exception(error)
  
    self.image_dtype = np.uint8
    print("--- num_classes {} image data_type {}".format(self.num_classes, self.image_dtype))
    print("--- num_images {} {} {}".format(num_images, self.image_height, self.image_width, self.image_channels))

    X = np.zeros((num_images, self.image_height, self.image_width, self.image_channels),
                 dtype=self.image_dtype)
       
    self.mask_dtype = bool
    if self.num_classes >1:
      self.mask_dtype = np.int8

    print("--- num_classes {} mask data_type  {}".format(self.num_classes, self.mask_dtype))
    Y = np.zeros((num_images, self.image_height, self.image_width, self.num_classes, ), 
                 dtype=self.mask_dtype)

    for n, image_file in enumerate(image_files):  
      img = cv2.imread(image_file)  
      if self.color_order == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
      img = cv2.resize(img, (self.image_width, self.image_height))
      X[n] = img

    for n, mask_file in enumerate(mask_files):
      if mask_file.endswith(".npz"):
        data = np.load(mask_file)
        mask = data['mask']
      elif mask_file.endswith(".png"):
        mask = self.rgb_to_categorized_mask(mask_file)
      Y[n] = mask
    return X, Y
    
  # Return PIL indexed mask
  def rgb_mask_to_indexed_mask(self, rgb_mask_file):
    rgb_mask = Image.open(rgb_mask_file).convert(self.color_order)
    if self.verbose:
       width, height = rgb_mask.size
       print("--rgb_mask_file {} image width {} height {}".format(rgb_mask_file, width, height))
    #2025/07/03 Added the following line
    rgb_mask = rgb_mask.resize((self.image_width, self.image_height))

    rgb_mask_array   = np.array(rgb_mask)
    indexed_array = np.argmin(np.sum((rgb_mask_array - self.rgb_array)**2, axis=-1), axis=0)

    # Create PIL image
    indexed_mask = Image.fromarray(indexed_array.astype(np.uint8), mode="P")
    indexed_mask.putpalette(self.palette)
    return indexed_mask
 
 
  # Convert an rgb mask to categorized_mask
  def rgb_to_categorized_mask(self, rgb_mask_file):
    # 1. Convert an rgb mask to an index mask
    # PIL indexed_mask will be returned
    indexed_mask = self.rgb_mask_to_indexed_mask(rgb_mask_file)
    
    # convert PIL image to nummpu array
    indexed_array = np.array(indexed_mask)
    if self.verbose:
      print("Shape of indexed array :", indexed_array.shape)
    # from tensorflow.keras.utils import to_categorical
    # 2. Create a categorized mask (numpy array) from an indexed mask
    categorized_mask = to_categorical(indexed_array, num_classes=self.num_classes)
    if self.verbose:
      print("Shape of categorized_mask:", categorized_mask.shape)
    return categorized_mask
    

  def to_categorized_mask(self, rgb_mask_array):
    indexed_array = np.argmin(np.sum((rgb_mask_array - self.rgb_array)**2, axis=-1), axis=0)

    # Create PIL image
    indexed_mask = Image.fromarray(indexed_array.astype(np.uint8), mode="P")
    if self.verbose:
      print("to_to_categorized_mask palette {}".format(self.palette))
    indexed_mask.putpalette(self.palette)
    indexed_array = np.array(indexed_mask)
    if self.verbose:
      print("to_to_categorized_mask palette num_classes {}".format(self.num_classes))
    # from tensorflow.keras.utils import to_categorical
    # 2. Create a categorized mask (numpy array) from an indexed mask
    categorized_mask = to_categorical(indexed_array, num_classes=self.num_classes)
    if self.verbose:
      print("Shape of categorized_mask:", categorized_mask.shape)
    return categorized_mask

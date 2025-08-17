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
# 2025/07/07 Updated
# ImageCategorizedMaskDatasetGenerator.py

import os
import shutil
import numpy as np
import cv2
import glob
import random

import traceback
from ConfigParser import ConfigParser
from ImageMaskAugmentor import ImageMaskAugmentor
from ImageCategorizedMaskDataset import ImageCategorizedMaskDataset

class ImageCategorizedMaskDatasetGenerator(ImageCategorizedMaskDataset):

  def __init__(self, config_file, dataset=ConfigParser.TRAIN, seed=137):
    super().__init__(config_file)
    random.seed = seed
    self.RGB            = "RGB"
    self.image_width    = self.config.get(ConfigParser.MODEL, "image_width")
    print("--- ImageCategorizedMaskDatasetGenerator ")

    self.image_height   = self.config.get(ConfigParser.MODEL, "image_height")
    self.image_channels = self.config.get(ConfigParser.MODEL, "image_channels")
    
    self.train_dataset  = [self.config.get(ConfigParser.TRAIN, "images_dir"),
                          self.config.get(ConfigParser.TRAIN, "masks_dir")]
    self.valid_dataset  = [self.config.get(ConfigParser.VALID, "images_dir"),
                          self.config.get(ConfigParser.VALID, "masks_dir")]
    self.batch_size     = self.config.get(ConfigParser.TRAIN, "batch_size")
    self.binarize       = self.config.get(ConfigParser.MASK, "binarize")
    self.threshold      = self.config.get(ConfigParser.MASK, "threshold")
    self.blur_mask      = self.config.get(ConfigParser.MASK, "blur")
    
  
    if not dataset in [ConfigParser.TRAIN, ConfigParser.VALID]:
      raise Exception("Invalid dataset")
      
    image_datapath = None
    mask_datapath  = None
  
    [image_datapath, mask_datapath] = self.train_dataset
    if dataset == ConfigParser.VALID:
      [image_datapath, mask_datapath] = self.valid_dataset
    image_files  = glob.glob(image_datapath + "/*.jpg")
    image_files += glob.glob(image_datapath + "/*.png")
    image_files += glob.glob(image_datapath + "/*.bmp")
    image_files += glob.glob(image_datapath + "/*.tif")
    image_files  = sorted(image_files)

    mask_files   = None
    if os.path.exists(mask_datapath):
      # PNG masks only
      mask_files = glob.glob(mask_datapath + "/*.png")
      mask_files = sorted(mask_files)
      
      if len(image_files) != len(mask_files):
        raise Exception("FATAL: Images and masks unmatched")
      
    num_images  = len(image_files)
    if num_images == 0:
      raise Exception("FATAL: Not found image files")
    
    self.image_datapath = image_datapath
    self.mask_datapath  = mask_datapath
    
    self.master_image_files = image_files
    self.master_mask_files  = mask_files
    self.augmentation       = self.config.get(ConfigParser.GENERATOR, "augmentation", dvalue=True)
    # We use ImageMaskAugmentor class to augment the orginal image and rgb_mask not categorized mask.
    self.image_mask_augmentor = ImageMaskAugmentor(config_file)

  def random_sampling(self, batch_size):
    if batch_size < len(self.master_image_files):
      images_sample = random.sample(self.master_image_files, batch_size)
    else:
      print("==- batch_size > the number of master_image_files")
      #if batch_size > the number of maste_image_files
      # we cannot apply random.sample function.
      #images_sample = random.sample(self.master_image_files, len_samples)
      images_sample = self.master_image_files
      # Force augmentation to be True
      self.augmentation = True
    
      self.image_mask_augmentor.rotation= True
      self.image_mask_augmentor.hflip   = True
      self.image_mask_augmentor.vflip   = True
      
    masks_sample  = []
    for image_file in images_sample:
      basename  = os.path.basename(image_file)
      name      = basename.split(".")[0]
      #mask file is PNG only
      mask_name = name + ".png"
      mask_file = os.path.join(self.mask_datapath, mask_name)
      if os.path.exists(mask_file):
        masks_sample.append(mask_file)
      else:
        raise Exception("Not found " + mask_file)
    images_sample = sorted(images_sample)
    masks_sample  = sorted(masks_sample)

    return (images_sample, masks_sample)

  # 
  def generate(self):
      print("---ImageCategorizedMaskDatasetGenerator.generate batch_size {}".format(self.batch_size))
      print("---  color_order {}".format(self.color_order))
      if self.color_order != self.RGB:
        raise Exception("Invalid color_order " + self.color_order)
      
      while True:
        (self.image_files, self.mask_files) = self.random_sampling(self.batch_size)

        IMAGES = []
        MASKS  = []
        for n, image_file in enumerate(self.image_files):
          mask_file = self.mask_files[n]

          image = cv2.imread(image_file)
          if self.color_order == self.RGB:
            image    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          else:
            raise Exception("Invalid image color_order")

          image = cv2.resize(image, dsize= (self.image_height, self.image_width), interpolation=cv2.INTER_NEAREST)
          IMAGES.append(image)
   
          mask    = cv2.imread(mask_file)
          if self.color_order == self.RGB:
            mask    = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
          else:
            raise Exception("Invalid mask color_order")
          mask    = cv2.resize(mask, dsize= (self.image_height, self.image_width), interpolation=cv2.INTER_NEAREST)
          MASKS.append(mask)
      
          if self.augmentation:
            #If augmentation enabled, we generate some augmented images and mask 
            # from the original image and mask. 
            self.image_mask_augmentor.augment(IMAGES, MASKS, image, mask)

        num_images = len(IMAGES)
        numbers = [i for i in range(num_images)]
        random.shuffle(numbers)
        
        if self.batch_size < num_images:
          target_numbers = random.sample(numbers, self.batch_size)
        else:
          target_numbers = numbers

        SELECTED_IMAGES = []
        SELECTED_MASKS  = [] 
        #print("--- target_numbers_len  {}  {}".format(len(target_numbers), target_numbers) )
        for i in target_numbers:
          SELECTED_IMAGES.append(IMAGES[i])
          # Here, we convert each rgb_mask in MASKS array to a categorized mask.
          # Convert rgb MASKS[i] to categorized CATMASK
          CATMASK = self.to_categorized_mask(MASKS[i])
          SELECTED_MASKS.append(CATMASK)

        # Convert the SELECTED list to numpy array.
        (X, Y) = self.convert(SELECTED_IMAGES, SELECTED_MASKS)
        yield (X, Y)

  def convert(self, IMAGES, MASKS):
    self.mask_dtype = bool
    if self.num_classes >1:
      self.mask_dtype = np.int8
    ilen = len(IMAGES)
    mlen = len(MASKS)
    X = np.zeros((ilen, self.image_height, self.image_width, self.image_channels), dtype=np.uint8)
    Y = np.zeros((mlen, self.image_height, self.image_width, self.num_classes, ),  dtype=self.mask_dtype)
    for i in range(ilen):
      X[i] = IMAGES[i]
      Y[i] = MASKS[i]
    return (X, Y)

  


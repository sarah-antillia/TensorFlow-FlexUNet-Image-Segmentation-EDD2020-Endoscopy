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
# EpochChangeInferencer.py
# 2025/06/05

import os
import glob
import cv2
import numpy as np
import shutil
from PIL import Image

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="true"

import tensorflow as tf


from ConfigParser import ConfigParser

class EpochChangeInferencer(tf.keras.callbacks.Callback):

  def __init__(self, flexmodel, config_file):
    self.flexmodel = flexmodel

    self.config = ConfigParser(config_file) 
    self.images_dir = self.config.get(ConfigParser.INFER,  "images_dir")
    self.output_dir = self.config.get(ConfigParser.TRAIN,  "epoch_change_infer_dir", dvalue="./epoch_change_infer")
    if os.path.exists(self.output_dir):
       shutil.rmtree(self.output_dir)
    os.makedirs(self.output_dir)

    self.num_infer_images = self.config.get(ConfigParser.TRAIN, "num_infer_images", dvalue=6)
    self.color_order = self.config.get(ConfigParser.MODEL,  "color_order", dvalue="RGB")
    sample_rgb_map   = {(0, 0, 0):0, (  0, 255,   0):1,  (255,   0,   0):2,  (0, 0, 255):3, }
    self.rgb_map =self.config.get(ConfigParser.MASK, "rgb_map", dvalue=sample_rgb_map)
    print("--- rgb_map {}".format(self.rgb_map))
   
    # Create a flattened palette 
    self.palette = []
    for item in self.rgb_map:
       self.palette += list(item)    
    
    if not os.path.exists(self.output_dir):
       os.makedirs(self.output_dir) 

    if not os.path.exists(self.images_dir):
      raise Exception("Not found " + self.images_dir)
    self.image_files  = glob.glob(self.images_dir + "/*.png")
    self.image_files += glob.glob(self.images_dir + "/*.jpg")
    self.image_files += glob.glob(self.images_dir + "/*.tif")
    self.image_files += glob.glob(self.images_dir + "/*.bmp")
    num_images = len(self.image_files)
    if self.num_infer_images > num_images:
        self.num_infer_images =  num_images
    if self.num_infer_images < 1:
        self.num_infer_images =  1
    self.image_files = self.image_files[:self.num_infer_images]

  def on_epoch_end(self, epoch, logs):
    for image_file in self.image_files:
      #print("--- on eoch_change end {}".format(image_file))
      predicted = self.flexmodel.predict(image_file)
      basename = os.path.basename(image_file)
      
      # 2025/06/09 Added the following 2 lines.
      if basename.endswith(".jpg"):
         basename = basename.replace(".jpg", ".png")
      filename = basename
      filename = "Epoch_" +str(epoch+1) + "_" + basename
      output_filepath = os.path.join(self.output_dir, filename)
  
      predicted.save(output_filepath)
      #print("Saved prediction {}".format(output_filepath))

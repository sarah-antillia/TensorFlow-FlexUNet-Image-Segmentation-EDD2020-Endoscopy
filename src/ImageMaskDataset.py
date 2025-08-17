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

# ImageMaskDataset.py

import os
import numpy as np
import cv2
from tqdm import tqdm
import glob
import traceback
import tensorflow as tf

from ConfigParser import ConfigParser

class ImageMaskDataset:

  def __init__(self, config_file, verbose=True):
    self.verbose = verbose

    self.config         = ConfigParser(config_file)
    self.image_width    = self.config.get(ConfigParser.MODEL, "image_width", dvalue=512)
    self.image_height   = self.config.get(ConfigParser.MODEL, "image_height", dvalue=512)
    self.image_channels = self.config.get(ConfigParser.MODEL, "image_channels", dvalue=3)
    self.num_classes    = self.config.get(ConfigParser.MODEL, "num_classes")

    self.color_order    = self.config.get(ConfigParser.IMAGE, "color_order", dvalue="RGB")
    self.mask_file_format = self.config.get(ConfigParser.MASK, "mask_file_format", dvalue=".png")

  # Define your own create method in a subclass derived from this class.
  def create(self, images_dir, masks_dir):
    X = None
    Y = None
    return X, Y
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
# TensorFlowFlexModel.py

# 2025/06/05 T.Arai
# 2025/06/15 Modified to support an online augmentation.

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="true"

import sys
import glob
import shutil
import cv2
import tensorflow
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from  tensorflow.keras.losses import SparseCategoricalCrossentropy
from  tensorflow.keras.metrics import SparseCategoricalAccuracy 

from EpochChangeCallback import EpochChangeCallback
from EpochChangeInferencer import EpochChangeInferencer
from ImageCategorizedMaskDataset import ImageCategorizedMaskDataset
from ImageCategorizedMaskDatasetGenerator import ImageCategorizedMaskDatasetGenerator

from dice_coef_multiclass import dice_coef_multiclass, dice_loss_multiclass
import json
import traceback
from PIL import Image

from ConfigParser import ConfigParser

tf.compat.v1.disable_eager_execution()

# Plesae define your own create_model method in a subclass derived from this class.

class TensorFlowFlexModel:
  SEED            = 137
  BEST_MODEL_FILE = "best_model.h5"
  HISTORY_JSON    = "history.json"

  def __init__(self, config_file):
    self.config_file    = config_file
    self.config         = ConfigParser(config_file)
    self.seed           = self.SEED
    self.verbose        = self.config.get(ConfigParser.DEBUG, "verbose")
    self.image_height   = self.config.get(ConfigParser.MODEL, "image_height")
    self.image_width    = self.config.get(ConfigParser.MODEL, "image_width")
    self.image_channels = self.config.get(ConfigParser.MODEL, "image_channels")

    self.num_classes    = self.config.get(ConfigParser.MODEL, "num_classes")
    self.color_order    = self.config.get(ConfigParser.IMAGE,  "color_order", dvalue="RGB")
    
    self.num_layers     = self.config.get(ConfigParser.MODEL, "num_layers", dvalue=6)
    self.base_filters   = self.config.get(ConfigParser.MODEL,  "base_filters", dvalue=16)
    

    #Multi-class NumPy NPZ (multi-class categorized mask file format) 
    numpy_npz = ".npz"
    self.mask_file_format = self.config.get(ConfigParser.MASK,  "mask_file_format", dvalue=numpy_npz)
        
    self.train_images_dir = self.config.get(ConfigParser.TRAIN, "images_dir")
    self.train_masks_dir  = self.config.get(ConfigParser.TRAIN, "masks_dir" ) 

    print("--- self.train.images {}".format(self.train_images_dir) )
    print("--- self.train.masks  {}".format(self.train_masks_dir) )
      
    self.valid_images_dir = self.config.get(ConfigParser.VALID, "images_dir")
    self.valid_masks_dir  = self.config.get(ConfigParser.VALID, "masks_dir" ) 
                                            
      
    Dataset     = eval(self.config.get(ConfigParser.DATASET, "class_name", dvalue="ImageCategorizedMaskDataset"))
    self.dataset = Dataset(config_file, verbose=self.verbose)
    print("--- Dataset class {}".format(self.dataset))

    self.mini_test_dir    = self.config.get(ConfigParser.INFER, "images_dir")

    self.mini_test_output_dir = self.config.get(ConfigParser.INFER, "output_dir")

    if os.path.exists(self.mini_test_output_dir):
      shutil.rmtree(self.mini_test_output_dir)
    os.makedirs(self.mini_test_output_dir)

    # 205/06/07
    # Default mask_datatype = "categorized",   mask_file_format = ".npz"
    # You may specify mask_datatype="indexed", mask_file_format = ".png"
    self.mask_datatype = self.config.get(ConfigParser.MASK, "mask_datatype", dvalue = "categorized")

    # Specify multi-class rgb color map dict as shown below
    sample_rgb_map = {(0, 0, 0):0, (  0, 255,   0):1,  (255,   0,   0):2,  (0, 0, 255):3, }
    self.rgb_map   = self.config.get(ConfigParser.MASK, "rgb_map", dvalue = sample_rgb_map)
    print("--- rgb_map {}".format(self.rgb_map))
   
    # Create a flattened palette from rgb_map
    self.palette = []
    for item in self.rgb_map:
       self.palette += list(item)
    # palette   = [0, 0, 0, 0, 255, 0, 255,0,0, 0, 0, 255]
    print("--- palette {}".format(self.palette))
    self.rev_rgb_map = {v: k for k, v in self.rgb_map.items()} 

    model_dir       = self.config.get(ConfigParser.TRAIN, "model_dir", dvalue="./models")
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)

    self.weight_filepath = os.path.join(model_dir, self.BEST_MODEL_FILE)

    sample_train_metrics = ["dice_coef_multiclass", "val_dice_coef_multiclass"]
    self.train_metrics  = self.config.get(ConfigParser.TRAIN, "metrics", dvalue=sample_train_metrics)

    self.model = self.create_model()
    if self.model == None:
      raise Exception("Failed to create a model")
    
    self.compile_model()

    self.model.loaded = False


  # Please define your own create_model method in a subclass
  def create_model(self):
    #raise Exception("Error: Not implemented yet")
    return None 
    
  def compile_model(self):
    if self.model == None:
      raise Exception("Not found model")
    
    learning_rate  = self.config.get(ConfigParser.MODEL, "learning_rate", dvalue=0.0001)
    clipvalue      = self.config.get(ConfigParser.MODEL, "clipvalue", dvalue=0.2)
    print("--- clipvalue {}".format(clipvalue))
  
    optimizer = self.config.get(ConfigParser.MODEL, "optimizer", dvalue="Adam")
    if optimizer == "Adam":
      self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate,
          beta_1    = 0.9, 
          beta_2    = 0.999, 
          clipvalue = clipvalue, 
          amsgrad   = False)
      print("--- Optimizer Adam learning_rate {} clipvalue {} ".format(learning_rate, clipvalue))
    
    elif optimizer == "AdamW":
      self.optimizer = tf.keras.optimizers.AdamW(learning_rate = learning_rate,
         clipvalue = clipvalue,)
      print("--- Optimizer AdamW learning_rate {} clipvalue {} ".format(learning_rate, clipvalue))

    # Specify a list of metrics name, which can be used to compile a UNet model. 
    metrics  = self.config.get(ConfigParser.MODEL, "metrics", dvalue=["dice_coef_multiclass"])
   
    self.metrics = []
    for metric in metrics:
      m = eval(metric)
      self.metrics.append(m)

    self.loss = self.config.get(ConfigParser.MODEL, "loss", dvalue="categorical_crossentropy") 

    self.model.compile(optimizer = self.optimizer,
              loss    = self.loss,
              metrics = self.metrics)

  # 2025/06/06: Fixed a bug.
  def create_callbacks(self):
    callbacks  = []
    check_point_callback     = tf.keras.callbacks.ModelCheckpoint(self.weight_filepath, verbose=1, 
                                     save_best_only    = True,
                                     save_weights_only = False)
    callbacks.append(check_point_callback)

    enable_reducer = self.config.get(ConfigParser.TRAIN, "learning_rate_reducer", dvalue=False)
    if enable_reducer:
      reducer_patience = self.config.get(ConfigParser.TRAIN, "reducer_patience", dvalue=4)
      reducer_factor   = self.config.get(ConfigParser.TRAIN, "reducer_factor", dvalue=0.4)
      reducer_callback = tf.keras.callbacks.ReduceLROnPlateau(
                        monitor = 'val_loss',
                        factor  = reducer_factor, 
                        reduer_patience = reducer_patience,
                        min_lr  = 0.0)
      print("--- Added reducer callback patience {}".format(reducer_patience))
      callbacks.append(reducer_callback)
  
    eval_dir = self.config.get(ConfigParser.EVAL, "eval_dir", dvalue="./eval")

    epoch_change_inferencer   = EpochChangeInferencer(self, self.config_file)
    callbacks.append(epoch_change_inferencer)

    epoch_change_callback   = EpochChangeCallback(eval_dir, self.train_metrics)
    callbacks.append(epoch_change_callback)

    # Patience to EarlyStopping callback.    
    # 2025/06/06 Moved here.
    patience = self.config.get(ConfigParser.TRAIN, "patience", dvalue=10)
    if patience > 0:
      early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=patience, verbose=1)
      print("--- Added earlyStopping callback patience {}".format(patience))
      callbacks.append(early_stopping_callback)

    print("--- Callbacks {}".format(callbacks))
    return callbacks

  def train(self):
    print("==== Started TensorFlowFlexModel.train")  
    eval_dir       = self.config.get(ConfigParser.TRAIN, "eval_dir", dvalue="./eval")
    #
    if not os.path.exists(eval_dir):
      os.makedirs(eval_dir)

    self.callbacks = self.create_callbacks()
    self.batch_size = self.config.get(ConfigParser.TRAIN, "batch_size", dvalue=4)
    self.epochs     = self.config.get(ConfigParser.TRAIN, "epochs",     dvalue=100)

    # Check generator (online_augmentation) flag
    generator = self.config.get(ConfigParser.MODEL, "generator", dvalue=False)
    #
    # 2025/06/15 Modified to suppor on line augmentation
    history  = None 
    if generator == False:

      X_train, Y_train = self.dataset.create(self.train_images_dir, self.train_masks_dir)
      X_valid, Y_valid = self.dataset.create(self.valid_images_dir, self.valid_masks_dir)
 
      print("=== Train data shape:", X_train.shape, Y_train.shape)
      print("=== Valid data shape:", X_valid.shape, Y_valid.shape)
      history = self.model.fit(X_train, Y_train,
                    validation_data = (X_valid, Y_valid),
                    callbacks       = self.callbacks,
                    batch_size      = self.batch_size,
                    epochs          = self.epochs,
                    verbose         = 1)

    else:
      # 2025/06/15 
      # On line augumentation
      # Create two  ImageCategorizedMaskDatasetGenerator for TRAIN and EVAL datasets.
      print("----train by generator")
      train_gen = ImageCategorizedMaskDatasetGenerator(self.config_file, dataset=ConfigParser.TRAIN)
      train_generator = train_gen.generate()
      valid_gen = ImageCategorizedMaskDatasetGenerator(self.config_file, dataset=ConfigParser.VALID)
      valid_generator = valid_gen.generate()

      history = self.train_by_generator(train_generator, valid_generator)

    self.save_history(history)

  # 2025/06/15
  def train_by_generator(self, train_generator, valid_generator):
    print("=== Start train_by_generator")
    print("--- Use the train and valid generator to fit.")
    # train and valid dataset will be used by train_generator and valid_generator respectively
    steps_per_epoch  = self.config.get(ConfigParser.TRAIN, "steps_per_epoch",  dvalue=400)
    validation_steps = self.config.get(ConfigParser.TRAIN, "validation_steps", dvalue=800)
  
    history = self.model.fit(train_generator, 
                    steps_per_epoch = steps_per_epoch,
                    epochs          = self.epochs, 
                    validation_data = valid_generator,
                    validation_steps= validation_steps,
                    shuffle         = False,
                    callbacks       = self.callbacks,
                    verbose         = 1)
    return history
  

  def save_history(self, history): 
    if history == None:
      return
    jstring = str(history.history)
    with open(self.HISTORY_JSON, 'wt') as f:
      json.dump(jstring, f, ensure_ascii=False,  indent=4, sort_keys=True, separators=(',', ': '))
      print("=== Saved {}".format(self.HISTORY_JSON))


  def evaluate(self):
    self.load_model()

    self.test_images_dir = self.config.get(ConfigParser.TEST, "images_dir",)
    self.test_masks_dir  = self.config.get(ConfigParser.TEST, "masks_dir",)
    X_test, Y_test = self.dataset.create(self.test_images_dir, self.test_masks_dir)
 
    batch_size = self.config.get(ConfigParser.EVAL, "batch_size", dvalue=4)
    scores = self.model.evaluate(X_test, Y_test, 
                                batch_size = batch_size,
                                verbose = 1)
    test_loss     = str(round(scores[0], 4))
    test_accuracy = str(round(scores[1], 4))
    print("Test loss    :{}".format(test_loss))     
    print("Test accuracy:{}".format(test_accuracy))
    #metrics = self.config.get(ConfigParser.MODEL, "metrics")
    
    loss    = self.config.get(ConfigParser.MODEL, "loss")
    metrics = self.config.get(ConfigParser.MODEL, "metrics")
    #metric = self.train_metrics[0]
    evaluation_result_csv = "./evaluation.csv"    
    with open(evaluation_result_csv, "w") as f:
      line = loss + "," + str(test_loss)
      f.writelines(line + "\n")     
      line = metrics[0] + "," + str(test_accuracy)
      f.writelines(line + "\n")     
 
    print("=== Saved {}".format(evaluation_result_csv))


  def infer(self):
    self.load_model()

    image_files  = glob.glob(self.mini_test_dir + "/*.png")
    image_files += glob.glob(self.mini_test_dir + "/*.jpg")
    image_files += glob.glob(self.mini_test_dir + "/*.tif")
    image_files += glob.glob(self.mini_test_dir + "/*.bmp")

    for image_file in image_files:
      predicted_rgb_mask = self.predict(image_file)
      basename = os.path.basename(image_file)
      #2025/06/10
      if basename.endswith(".jpg"):
        basename = basename.replace(".jpg", ".png")
      output_filepath = os.path.join(self.mini_test_output_dir, basename)
      basename = os.path.basename(image_file)
    
      predicted_rgb_mask.save(output_filepath)
      print("=== Saved prediction {}".format(output_filepath))


  def predict(self, image_file):
      img = cv2.imread(image_file)
      if self.color_order == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = cv2.resize(img, (self.image_width, self.image_height))

      img = np.expand_dims(img, axis=0)
      predicted = self.model.predict(img)
      predicted_argmax = np.argmax(predicted, axis=-1)[0] 
 
      predicted_rgb_mask = self.convert_to_rgb(predicted_argmax)
      return predicted_rgb_mask
  
  def convert_to_rgb(self, prediction_array):
     img = Image.fromarray(prediction_array.astype(np.uint8))
     img = img.convert('P')
     img.putpalette(self.palette)
     return img


  def load_model(self) :

    if os.path.exists(self.weight_filepath):
      if self.model == None:
        raise Exception("Not found model")
      if self.model.loaded == False:
        self.model.load_weights(self.weight_filepath)
        print("=== Loaded a weight_file {}".format(self.weight_filepath))
        self.model.loaded = True

    else:
      error = "Not found a weight_file " + self.weight_filepath
      raise Exception(error)


def inspect(self, image_file='./model.png', summary_file="./summary.txt"):
    # Please download and install graphviz for your OS
    # https://www.graphviz.org/download/ 
    tf.keras.utils.plot_model(self.model, to_file=image_file, show_shapes=True)
    print("=== Saved model graph as an image_file {}".format(image_file))
    # https://stackoverflow.com/questions/41665799/keras-model-summary-object-to-string
    with open(summary_file, 'w') as f:
      # Pass the file handle in as a lambda function to make it callable
      self.model.summary(print_fn=lambda x: f.write(x + '\n'))
    print("=== Saved model summary as a text_file {}".format(summary_file))


if __name__ == "__main__":
  try:
    config_file = "./train_eval_infer.config"
    if len(sys.argv) == 2:
      config_file = sys.argv[1]
    if not os.path.exists(config_file):
      raise Exception("No found " + config_file)
    model = TensorFlowFlexModel(config_file)

  except:
    traceback.print_exc()


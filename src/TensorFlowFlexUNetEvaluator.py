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

# 2025/06/05
# TensorFlowFlexUNetInferencer.py

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="true"

import sys
import traceback

from ConfigParser import ConfigParser

# 205/07/09 Added the following classes.

from TensorFlowFlexUNet          import TensorFlowFlexUNet
from TensorFlowFlexAttentionUNet import TensorFlowFlexAttentionUNet
from TensorFlowFlexDeepLabV3Plus import TensorFlowFlexDeepLabV3Plus

from TensorFlowFlexSwinUNet      import TensorFlowFlexSwinUNet
from TensorflowFlexEfficientNetB7UNet import TensorFlowFlexEfficientNetB7UNet
from TensorflowFlexEfficientUNet import TensorFlowFlexEfficientUNet
from TensorFlowFlexMultiResUNet  import TensorFlowFlexMultiResUNet
from TensorFlowFlexSharpUNet     import TensorFlowFlexSharpUNet
from TensorFlowFlexU2Net         import TensorFlowFlexU2Net
from TensorFlowFlexUNet3Plus     import TensorFlowFlexUNet3Plus
from TensorFlowFlexTransUNet     import TensorFlowFlexTransUNet

if __name__ == "__main__":
  try:
    config_file = "./train_eval_infer.config"
    if len(sys.argv) == 2:
      config_file = sys.argv[1]
    if not os.path.exists(config_file):
      raise Exception("Not found " + config_file)
    config = ConfigParser(config_file)

    MODEL_CLASS = eval(config.get(ConfigParser.MODEL, "model"))
    print("---MODEL_CLASS {}".format(MODEL_CLASS))
    model = MODEL_CLASS(config_file)
    
    model.load_model()
    model.evaluate()

  except:
    traceback.print_exc()


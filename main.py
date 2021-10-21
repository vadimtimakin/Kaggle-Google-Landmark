import warnings
warnings.filterwarnings("ignore")

from config import config

from utils import set_seed, classic_training, optimize_model
from kd_training import kd_training
from val_inference import val_inference
from inference_utils import inference

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # RTX 3080 with cuda-11 fix

def run(config):
    set_seed(seed=config.general.seed)

    mode = config.general.mode

    if mode == "classic_training":
        classic_training(config)  # Single model training
    elif mode == "kd_training":
        kd_training(config)       # Knowledge Distillation (teacher-student) training
    elif mode == "val_inference":
        val_inference(config)     # Inference and evaluating on validation dataset
    elif mode == "inference":
        inference(config)         # Inference on test dataset
    elif mode == "optimize_model":
        optimize_model(config)    # Optimizing model (fusion, quantization, torch.jit.script)
    else:
        raise ValueError("Invalid session mode.")

# First run in the debug mode
print("Running in debug mode")
config.training.number_of_debug_samples = 1000
config.data.kfold.use_kfold = True
config.training.debug = True
run(config)

# Then run the full training
print("Running the full training")
config.data.kfold.use_kfold = False
config.training.debug = False
run(config)
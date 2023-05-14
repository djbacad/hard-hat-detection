# CLI commands:
# Change directory to src: 
## cd src

# Run the script and write logs to tuning_output txt file: 
## python tune.py > tuning_output.txt

import torch
import os
from ultralytics import YOLO
from ray import tune

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

current_dir = os.getcwd()
relative_path = 'Hard-Hat-Workers-8\custom_hard_hat.yaml'
dataset_path = os.path.abspath(os.path.join(current_dir, 
                                            os.pardir, 
                                            relative_path))

# Load the model
model = YOLO("yolov8n.pt")

# Hyperparameter Tuning with Ray Tune demonstration:
# The model class of yolov8 utilizes Ray Tune's ASHA scheduler.
# There are constraints in resources specifically in computing power so the search space 
# has been limited to the initial learning rate, weight decay, momentum and rotation augmentation range.
# The tuning process is also only set to run for 5 trials, 10 epochs each.
# Review all tunable hyperparameters for yolov8 architecture here: 
# https://docs.ultralytics.com/usage/hyperparameter_tuning/?h=tune
result = model.tune(
    data=dataset_path,
    gpu_per_trial=1,
    max_samples=5,
    space={
      "lr0": tune.uniform(1e-5, 1e-2),
      "weight_decay": tune.uniform(0.0, 0.001),
      "momentum": tune.uniform(0.6, 0.7),
      "degrees": tune.uniform(0.0, 15.0)
      },
    train_args={
      "epochs":10, 
      "imgsz":640, 
      "batch":32
      }
)

# Get best params (maximum mAPs):
best_params_mAP50 = result.get_best_result(metric='metrics/mAP50(B)', mode='max')
best_params_mAP50_95 = result.get_best_result(metric='metrics/mAP50(B)', mode='max')

print(f'Max mAP50: {best_params_mAP50}')
print(f'Max mAP50-95: {best_params_mAP50_95}')
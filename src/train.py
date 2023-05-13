from ultralytics import YOLO
from ray import tune

# Load the model
model = YOLO("yolov8n.pt")

# Hyperparameter Tuning with Ray Tune:
result = model.tune(
    data='E:/files_main/projects_ml/hard-hat-detection/Hard-Hat-Workers-8/custom_hard_hat.yaml',
    gpu_per_trial=1,
    max_samples=5,
    space={"lr0": tune.uniform(1e-5, 1e-1),
          "weight_decay": tune.uniform(0.0, 0.001),
          "momentum": tune.uniform(0.6, 0.7),
          "degrees": tune.uniform(0.0, 15.0)
          },
    train_args={"epochs": 10, 
                "imgsz":640, 
                "batch":8}
)
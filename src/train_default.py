# CLI commands:
# Change directory to src: 
## cd src

# Run the script and write output to training_output txt file: 
## python train_default.py > training_output.txt

import torch
import os
from ultralytics import YOLO

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    current_dir = os.getcwd()
    relative_path = 'Hard-Hat-Workers-8\custom_hard_hat.yaml'
    dataset_path = os.path.abspath(os.path.join(current_dir, 
                                                os.pardir, 
                                                relative_path))

    # Load the model.
    model = YOLO('yolov8n.pt')
    
    # Train using default hyperparameters:
    results = model.train(
    data=dataset_path,
    imgsz=640,
    epochs=10,
    batch=32,
    name='yolov8n_custom_default'
    )


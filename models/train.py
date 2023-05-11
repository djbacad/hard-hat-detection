from ultralytics import YOLO
import torch
import multiprocessing

def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    run()

# Load the model.
model = YOLO('yolov8n.pt')
 
# Training.
results = model.train(
   data='../Hard-Hat-Workers-8/custom_hard_hat.yaml',
   imgsz=640,
   epochs=10,
   batch=8,
   name='yolov8n_custom'
   )
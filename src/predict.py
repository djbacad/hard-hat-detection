import argparse
import cv2
from ultralytics import YOLO

# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("video_name", help="path to the input video file")
args = parser.parse_args()

# Load the YOLOv8 model
model = YOLO('../ultralytics/runs/detect/yolov8n_custom_default/weights/best.pt')

# Open the video file
video_path = f'../test/{args.video_name}'
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("Custom YOLOv8 Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
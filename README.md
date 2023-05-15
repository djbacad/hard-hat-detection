Real-time Hard-hat Detection
==================================
![gif_final2](https://github.com/djbacad/hard-hat-detection/assets/61301478/7501b4ea-a610-4310-94c2-2c65e775f77f)
![gif_final1](https://github.com/djbacad/hard-hat-detection/assets/61301478/88b29c52-e07c-4e66-abb3-3ca3b0fb48cb)

In this project, a customized object detection model for hard-hats was built using the YOLOv8nano architecture and tuned using the Ray Tune hyperparameter tuning framework.

### Project Highlights:
- The dataset used is publicly available in [Roboflow CV Datasets](https://public.roboflow.com/). Signing up / creating an account in Roboflow provides convenience in downloading datasets and exporting annotations in formats that are YOLO-ready. The download.py script can serve as a reference on how to download datasets in Roboflow using an API key.
- Rather than using default hyperparameters for building the custom [YOLOv8](https://github.com/ultralytics/ultralytics) object detection model, the best set of hyperparameters were searched for using the [Ray Project](https://www.ray.io/)'s hyperparameter tuning capability to achieve better mAP50.

### Hardware:
- Nvidia GeForce RTX 2060 Mobile GPU
- Ryzen 7 4800H CPU

### Operating System:
- Windows 11

### Limitations:
- Due to computing power constraints, the search space for the hyperparameter tuning process were limited to only the initial learning rate, weight decay, momentum, and rotation augmentation ranges.

### Future Work:
- Expand the search space and experiment on using larger image sizes.
- Add additional classes like safety glasses, hi-vis vests, safety shoes, and other safety wearables to build a complete safety surveillance solution for construction sites.

### Try the code:

Place the test video file inside test folder, navigate inside the src folder and issue the ff command:
```cmd
python predict.py <your_video_filename>
```

### Credits/Citations:

Jocher, G., Chaurasia, A., & Qiu, J. (2023). YOLO by Ultralytics (Version 8.0.0) [Computer software]. https://github.com/ultralytics/ultralytics

Ray Project. (2023). ray-project/ray (Version 2.4.0). GitHub. https://github.com/ray-project/ray

from roboflow import Roboflow
rf = Roboflow(api_key=<YOUR_API_KEY>)
project = rf.workspace("joseph-nelson").project("hard-hat-workers")
dataset = project.version(8).download("yolov8")
from roboflow import Roboflow
rf = Roboflow(api_key="2hudbTMMi5m0eOeEgmmQ")
project = rf.workspace("joseph-nelson").project("hard-hat-workers")
dataset = project.version(8).download("yolov8")

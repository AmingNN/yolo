#!/usr/bin/python
# -*- coding:utf-8 -*-
# FileName  :predict.py
# Time      :2024/6/26 上午1:28
# Author    :Aming
# desc      :None

from ultralytics import YOLO

# model = YOLO("/home/aming/project/AI/ultralytics/ultralytics/cfg/models/v9/yolov9c.yaml").load("/home/aming/project/AI/AIInspection/yolov9_2000_exp/yolov9.pt")
model = YOLO("runs/detect/train4/weights/best.pt")
# model = YOLO("ultralytics/cfg/models/v5/yolov5.yaml")

# model.train(data="/home/aming/project/AI/Datasets/YOLO/PCB/pcb.yaml", epochs=1, batch=4)

results = model("/home/aming/project/AI/AIInspection/yolov9_2000_exp/01_mouse_bite_01.jpg")
print(results)
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    print(boxes, masks, keypoints, probs, obb)


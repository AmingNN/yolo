#!/usr/bin/python
# -*- coding:utf-8 -*-
# FileName  :detect.py
# Time      :2024/6/26 上午1:59
# Author    :Aming
# desc      :None

from ultralytics import YOLO

data = "../Datasets/YOLO/PCB/pcb.yaml"
model_yaml = "ultralytics/cfg/models/v9/yolov9c.yaml"
model_path = ""

if model_yaml and model_path:
    model = YOLO(model_yaml).load(model_path)
elif model_yaml and not model_path:
    model = YOLO(model_yaml)
else:
    model = YOLO(model_path)


model.train(data=data, epochs=1, batch=4)

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
print(f"source: {metrics.box.map}")  # map50-95



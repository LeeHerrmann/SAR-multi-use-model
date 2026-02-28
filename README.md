# eVident – People detection from drones using YOLOv8

[![License:Apache 2.0]]
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/model-YOLOv8s-green)](https://github.com/ultralytics/ultralytics)

**eVident** is a computer vision model for detecting people in drone footage.  
It was trained on images from forests, fields, and urban areas at various altitudes. You can fine-tune it for your own use case.

![demo](assets/demo.gif)  
*Example: model detecting people in a drone frame.*

---

## Features

- Detects people in drone images and videos.
- Runs in real time (~30 fps on Jetson Nano, ~15 fps on a typical CPU).
- Exportable to ONNX, TensorRT, OpenVINO.
- Open source, feel free to modify.

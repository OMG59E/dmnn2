'''
Author: xingwg
Date: 2024-12-13 17:17:01
LastEditTime: 2024-12-15 12:47:45
FilePath: /dmnn2/models/yolo_world_v2/add_EfficientNMS_TRT.py
Description: 

Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
'''
import sys
import numpy as np
import onnx
import onnx_graphsurgeon as gs

onnx_file = sys.argv[1]
new_file = sys.argv[2]

graph = gs.import_onnx(onnx.load(onnx_file))

score_threshold = 0.01
num_detections = 100
iou_threshold = 0.7

nms_attrs = {
    "plugin_version": "1",
    "background_class": -1,
    "max_output_boxes": num_detections,
    "score_threshold": max(0.01, score_threshold),  # Keep threshold to at least 0.01 for better efficiency
    "iou_threshold": iou_threshold,
    "score_activation": 0,
    "class_agnostic": 1,
    "box_coding": 0,
}

nms_output_classes_dtype = np.int32
nms_output_num_detections = gs.Variable(name="num_detections", dtype=np.int32, shape=[1, 1])
nms_output_boxes = gs.Variable(name="detection_boxes", dtype=np.float32, shape=[1, num_detections, 4])
nms_output_scores = gs.Variable(name="detection_scores", dtype=np.float32, shape=[1, num_detections])
nms_output_classes = gs.Variable(name="detection_classes", dtype=nms_output_classes_dtype, shape=[1, num_detections])
nms_outputs = [nms_output_num_detections, nms_output_boxes, nms_output_scores, nms_output_classes]

scores, bboxes = graph.outputs
nms_op = gs.Node(
    op="EfficientNMS_TRT",
    name="EfficientNMS_TRT",
    attrs=nms_attrs,
    inputs=[bboxes, scores],
    outputs=nms_outputs
)

graph.nodes.append(nms_op)

graph.outputs = nms_outputs

graph.cleanup().toposort()

onnx.save(gs.export_onnx(graph), new_file)



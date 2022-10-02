# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.append('.')
sys.path.append('./yolov5')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams, VID_FORMATS
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
from yolox.utils.visualize import plot_tracking

# model --> detector
# tracker --> tracker



def detect_track(image, model, tracker):

    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size( (640, 640), s=stride)  # check image size

    # 外部创建跟踪器
    # Create tracker
    # tracker = BYTETracker(0.5, 30, 0.8, False, 30)


    device = select_device(0)
    half = True

    online_tlwhs = []
    online_ids = []
    online_scores = []

    # Run tracking
    im = torch.from_numpy(image).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    pred = model(im, augment=False, visualize=False)
    # Apply NMS
    pred = non_max_suppression(pred, 0.5, 0.5, None, False, max_det=1000)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], image.shape).round()

            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]
        
            online_targets  = tracker.update(det[:, :5].cpu(), [im0.shape[0], im0.shape[1]], imgsz)
            
            if len(online_targets) > 0:
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
    return online_tlwhs, online_ids, online_scores
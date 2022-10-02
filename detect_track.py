import os
import sys
par_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(par_path)
sys.path.append(os.path.join(par_path, 'yolov5'))

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

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.augmentations import letterbox
from tracker.byte_tracker import BYTETracker
from deepsort_tracker.deepsort import Tracker,NearestNeighborDistanceMetric
from sort_tracker.sort import Sort

import warnings
warnings.filterwarnings("ignore")



class Detect_Track:
    def __init__(self, tracker='ByteTracker', model_path='./models/yolov5s.pt', imgsz=(640,640), vis=True):
        yolo_model = os.path.join(par_path, model_path)
        
        self.device = torch.device(0)
        self.model = DetectMultiBackend(yolo_model, device=self.device, dnn=False, fp16=True)

        self.names = self.model.names
        self.stride = self.model.stride
        self.imgsz = check_img_size( imgsz, s=self.stride)  # check image size
        
        self.model.warmup(imgsz=(1, 3, *self.imgsz))

        self.trt = model_path.endswith('.trt')

        self.vis = vis

        
        if tracker == 'ByteTracker':
            self.tracker = BYTETracker(0.5, 70, 0.8, False, 30)
            self._type = 'ByteTracker'
        elif tracker == 'Deepsort':
            max_cosine_distance = 0.1
            metric = NearestNeighborDistanceMetric(
                "cosine", max_cosine_distance, 100)
            self.tracker = Tracker(
                metric, max_iou_distance=0.7, max_age=30, n_init=3)
            self._type = 'DeepSort'
        elif tracker == 'Sort':
            self.tracker = Sort(0.7)
            self._type = "Sort"
        else:
            raise Exception('Tracker must be ByteTracker/Deepsort/Sort.')

        
    @torch.no_grad()
    def __call__(self, image: np.ndarray):

        img_vis = image.copy()
        
        clss = []
        tlwhs = []
        tids = []

        # Run tracking
        img = letterbox(image, self.imgsz, stride=self.stride, auto= not self.trt)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        im = img.half()
        im /= 255.0
        if len(im.shape) == 3:
            im = im[None]
        
        # inference
        pred = self.model(im, augment=False, visualize=False)
        # Apply NMS
        pred = non_max_suppression(pred, 0.5, 0.5, None, False, max_det=1000)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], image.shape).round()
                
                if self._type == 'ByteTracker':
                    online_targets  = self.tracker.update(det[:, :6].cpu(), [image.shape[0], image.shape[1]], self.imgsz)
                    
                    if len(online_targets) > 0:
                        for t in online_targets:

                            clss.append(t.cls.item())
                            tlwhs.append([ t.tlwh[0], t.tlwh[1], t.tlwh[2], t.tlwh[3] ])
                            tids.append(t.track_id)

                            if self.vis:
                                cv2.rectangle(img_vis, (int(t.tlwh[0]), int(t.tlwh[1])), (int(t.tlwh[0]+t.tlwh[2]), int(t.tlwh[1]+t.tlwh[3])), (255,0,0), 2)
                                cv2.putText(img_vis, self.names[int(t.cls.item())]+'  '+str(t.track_id), (int(t.tlwh[0]),int(t.tlwh[1])), cv2.FONT_HERSHEY_COMPLEX, 0.8, [0, 0, 255])
                
                elif self._type == 'Sort':
                    online_targets  = self.tracker.update(det[:, :6].cpu(), [image.shape[0], image.shape[1]], self.imgsz)
                    if len(online_targets) > 0:
                        for t in online_targets:
                            tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                            tid = t[4]

                            tlwhs.append(tlwh)
                            tids.append(int(tid))

                            if self.vis:
                                cv2.rectangle(img_vis, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0]+tlwh[2]), int(tlwh[1]+tlwh[3])), (255,0,0), 2)
                                cv2.putText(img_vis, str(tid), (int(t.tlwh[0]),int(t.tlwh[1])), cv2.FONT_HERSHEY_COMPLEX, 0.8, [0, 0, 255]) # self.names[int(t.cls.item())]+'  '+


        return img_vis, clss, tlwhs, tids


if __name__ =='__main__':
    video_path = './video/palace.mp4'
    capture = cv2.VideoCapture(video_path)
    model = Detect_Track()
    frame_id = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        img_vis, clss, tlwhs, tids = model(frame)
        if frame_id % 100 == 0:
            cv2.imwrite('./assert/{}.jpg'.format(frame_id), img_vis)
        frame_id += 1
        # cv2.imshow('track', img)
        # cv2.waitKey(10)

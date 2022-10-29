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
from bytetrack_tracker.byte_tracker import BYTETracker
from deepsort_tracker.deepsort import Tracker, NearestNeighborDistanceMetric
from sort_tracker.sort import Sort
from bot_tracker.mc_bot_sort import BoTSORT

import warnings
warnings.filterwarnings("ignore")

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


class Detect_Track:
    def __init__(self, tracker='ByteTrack', model_path='./models/yolov5s.pt', imgsz=(640,640), vis=True):
        yolo_model = os.path.join(par_path, model_path)
        
        self.device = torch.device(0)
        self.model = DetectMultiBackend(yolo_model, device=self.device, dnn=False, fp16=True)

        self.names = self.model.names
        self.stride = self.model.stride
        self.imgsz = check_img_size( imgsz, s=self.stride)  # check image size

        self.trt = model_path.endswith('.engine') or model_path.endswith('.trt')

        self.vis = vis

        
        if tracker == 'ByteTrack':
            self.tracker = BYTETracker(track_thresh=0.5, track_buffer=70, match_thresh=0.8)
            self._type = 'ByteTrack'

        elif tracker == 'Deepsort':
            max_cosine_distance = 0.1
            metric = NearestNeighborDistanceMetric(
                "cosine", max_cosine_distance, 100)
            self.tracker = Tracker(
                metric, max_iou_distance=0.7, max_age=30, n_init=3)
            self._type = 'DeepSort'

        elif tracker == 'Sort':
            self.tracker = Sort(det_thresh = 0.7)
            self._type = "Sort"

        elif tracker == "BoTSort":
            self.tracker = BoTSORT(track_high_thresh=0.3, track_low_thresh=0.05, new_track_thresh=0.4, 
                                   match_thresh=0.7, track_buffer=30,frame_rate=30,
                                   with_reid = False, proximity_thresh=0.5, appearance_thresh=0.25,
                                   fast_reid_config=None, fast_reid_weights=None, device=None)
            self._type = "BoTSort"

        else:
            raise Exception('Tracker must be ByteTrack/Deepsort/Sort/BoTSort.')

        
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
                
                if self._type == 'ByteTrack':
                    online_targets  = self.tracker.update(det[:, :6].cpu(), [image.shape[0], image.shape[1]], self.imgsz)
                    
                    if len(online_targets) > 0:
                        for t in online_targets:

                            clss.append(t.cls.item())
                            tlwhs.append([ t.tlwh[0], t.tlwh[1], t.tlwh[2], t.tlwh[3] ])
                            tids.append(t.track_id)

                            if self.vis:
                                color = get_color(int(t.cls.item())+1)
                                cv2.rectangle(img_vis, (int(t.tlwh[0]), int(t.tlwh[1])), (int(t.tlwh[0]+t.tlwh[2]), int(t.tlwh[1]+t.tlwh[3])), color, 2)
                                cv2.putText(img_vis, self.names[int(t.cls.item())]+'  '+str(t.track_id), (int(t.tlwh[0]),int(t.tlwh[1])), cv2.FONT_HERSHEY_COMPLEX, 0.8, color)
                
                elif self._type == 'Sort':
                    online_targets  = self.tracker.update(det[:, :6].cpu(), [image.shape[0], image.shape[1]], self.imgsz)
                    if len(online_targets) > 0:
                        for t in online_targets:
                            tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                            tid = t[4]
                            cls = t[5]
                            
                            tlwhs.append(tlwh)
                            tids.append(int(tid))
                            clss.append(cls)

                            if self.vis:
                                color = get_color(int(cls)+1)
                                cv2.rectangle(img_vis, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0]+tlwh[2]), int(tlwh[1]+tlwh[3])), color, 2)
                                cv2.putText(img_vis,  self.names[int(cls)]+'  '+str(int(tid)), (int(tlwh[0]),int(tlwh[1])), cv2.FONT_HERSHEY_COMPLEX, 0.8, color)

                elif self._type == "BoTSort":
                    online_targets  = self.tracker.update(det[:, :6].cpu(), image)
                    for t in online_targets:
                        tlwh = t.tlwh
                        tlbr = t.tlbr
                        tid = t.track_id
                        tcls = t.cls
                        
                        tlwhs.append(tlwh)
                        tids.append(int(tid))
                        clss.append(tcls)
                        
                        if self.vis:
                            color = get_color(int(tcls)+1)
                            cv2.rectangle(img_vis, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0]+tlwh[2]), int(tlwh[1]+tlwh[3])), color, 2)
                            cv2.putText(img_vis,  self.names[int(tcls)]+'  '+str(int(tid)), (int(tlwh[0]),int(tlwh[1])), cv2.FONT_HERSHEY_COMPLEX, 0.8, color) 

        return img_vis, clss, tlwhs, tids


if __name__ =='__main__':
    video_path = './video/palace.mp4'
    capture = cv2.VideoCapture(video_path)
    
    model = Detect_Track()
    frame_id = 0

    fps = capture.get(cv2.CAP_PROP_FPS) 
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
    outVideo = cv2.VideoWriter('./video/result.avi',fourcc,fps,size) 

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        img_vis, clss, tlwhs, tids = model(frame)
        outVideo.write(img_vis)
        frame_id += 1
        # cv2.imshow('track', img)
        # cv2.waitKey(10)
    capture.release() 
    outVideo.release() 

    from moviepy.editor import *

    clip = (VideoFileClip('./video/result.avi').subclip(1,5).resize(0.8))
    clip.write_gif("./assert/demo.gif")
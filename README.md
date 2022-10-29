# YoloV5_MCMOT
> YoloV5+sort/deepsort/bytetrack/BoTSort 多类别
> 尽量简单的调用

## Yolov5 + ByteTrack MCMOT 2022/10/2
- Detect_Track
```python
from detect_track import Detect_Track
model = Detect_Track(tracker='ByteTracker', model_path='./models/yolov5s.pt', 
                    imgsz=(640,640), vis=True)
img_vis, clss, tlwhs, tids = model(img)
```

## Demo
![image](./assert/demo.gif)

## Yolov5 + Sort MCMOT 2022/10/3
- Detect_Track
```python
from detect_track import Detect_Track
model = Detect_Track(tracker='Sort', model_path='./models/yolov5s.pt', 
                    imgsz=(640,640), vis=True)
img_vis, clss, tlwhs, tids = model(img)  
```

## Demo
![image](./assert/car_demo.gif)

## Yolov5 + BoTSort MCMOT 2022/10/29
- Detect_Track
```python
from detect_track import Detect_Track
model = Detect_Track(tracker='BoTSort', model_path='./models/yolov5s.pt', 
                    imgsz=(640,640), vis=True)
img_vis, clss, tlwhs, tids = model(img)  
```
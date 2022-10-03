# YoloV5_MCMOT
YoloV5+sort/deepsort/bytetrack+多类别

## Yolov5 + ByteTrack MCMOT 2022/10/2
- Detect_Track
```python
from detect_track import Detect_Track
model = Detect_Track(tracker='ByteTracker', model_path='./models/yolov5s.pt', 
                    imgsz=(640,640), vis=True)
img_vis, clss, tlwhs, tids = model(img)
```

## Yolov5 + Sort MOT 2022/10/2
- Detect_Track
```python
from detect_track import Detect_Track
model = Detect_Track(tracker='Sort', model_path='./models/yolov5s.pt', 
                    imgsz=(640,640), vis=True)
img_vis, clss, tlwhs, tids = model(img)  
# clss is None
```

## Demo
![image](./assert/demo.gif)

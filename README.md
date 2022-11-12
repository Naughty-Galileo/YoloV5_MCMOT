# YoloV5_MCMOT
> YoloV5+Sort/DeepSort/ByteTrack/BoTSort/motdt 多类别
> 尽量简单的调用 \
> tracker的参数从detect_track.py中更改

## Yolov5 + ByteTrack MCMOT 2022/10/2
```python
from detect_track import Detect_Track
model = Detect_Track(tracker='ByteTrack', model_path='./models/yolov5s.pt', 
                    imgsz=(640,640), vis=True)
img_vis, clss, tlwhs, tids = model(img)
```

## Demo
![image](./assert/demo.gif)

## Yolov5 + Sort MCMOT 2022/10/3
```python
from detect_track import Detect_Track
model = Detect_Track(tracker='Sort', model_path='./models/yolov5s.pt', 
                    imgsz=(640,640), vis=True)
img_vis, clss, tlwhs, tids = model(img)  
```

## Demo
![image](./assert/car_demo.gif)

## Yolov5 + BoTSort MCMOT 2022/10/29
```python
from detect_track import Detect_Track
model = Detect_Track(tracker='BoTSort', model_path='./models/yolov5s.pt', 
                    imgsz=(640,640), vis=True)
img_vis, clss, tlwhs, tids = model(img)  
```

## Yolov5 + DeepSort MCMOT 2022/11/12
```python
from detect_track import Detect_Track
model = Detect_Track(tracker='DeepSort', model_path='./models/yolov5s.pt', 
                    imgsz=(640,640), vis=True)
img_vis, clss, tlwhs, tids = model(img)  
```


## Yolov5 + MotDt MCMOT 2022/11/12
```python
from detect_track import Detect_Track
model = Detect_Track(tracker='motdt', model_path='./models/yolov5s.pt', 
                    imgsz=(640,640), vis=True)
img_vis, clss, tlwhs, tids = model(img)  
```
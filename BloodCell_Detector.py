import cv2
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import check_img_size,non_max_suppression,scale_coords
from yolov5.utils.torch_utils import select_device
from yolov5.utils.datasets import LoadImages , LoadStreams
from yolov5.utils.plots import plot_one_box
import numpy as np
import torch
from time import sleep

img_path = "Data/train/images"
# vid_path = "D:\Mask_Video"
model_path = "BloodCell-weights.pt"

device = select_device("cpu")
model = attempt_load(model_path,map_location=device)

stride = int(model.stride.max())
names = model.module.names if hasattr(model, 'module') else model.names

colors = [(0,69,255) , (80,198,107) ,  (0,165,255)]

Images = LoadImages(img_path, img_size=640, stride=stride)
# Images = LoadStreams("0", img_size=imgsz, stride=stride)  For Webcam

TotalBloodCellDetection = {'platelets' : 0 , 'RBC' : 0 , 'WBC' : 0}

for path, img, im0s, vid_cap in Images:  # for each image in Dir
    Draw_img = im0s
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img = img / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    Detection = model(img,False)[0]
    Detection = non_max_suppression(Detection,conf_thres=0.25,iou_thres=0.55, max_det=500)

    BloodCellDetection = {'platelets' : 0 , 'RBC' : 0 , 'WBC' : 0}

    for i, det in enumerate(Detection):  # detections per image

        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], Draw_img.shape).round()
            for *coor , conf , cls in reversed(det):
                plot_one_box(coor,Draw_img,colors[int(cls)],label=names[int(cls)],line_thickness=2)

                if cls==0:BloodCellDetection['platelets'] += 1
                elif cls==1:BloodCellDetection['RBC'] += 1
                elif cls==2:BloodCellDetection['WBC'] += 1

    screen = np.ones((90,600))
    BloodCellsStr = f"RBC : {BloodCellDetection['RBC']} , WBC : {BloodCellDetection['WBC']} , Platelets : {BloodCellDetection['platelets']}"
    TotalBloodCellsStr =  f"Total Blood Cells - RBC : {TotalBloodCellDetection['RBC']} , WBC : {TotalBloodCellDetection['WBC']} , Platelets : {TotalBloodCellDetection['platelets']}"
    cv2.putText(Draw_img,BloodCellsStr,(340,460),cv2.FONT_HERSHEY_TRIPLEX , 0.5 , (0,0,255),thickness=1)
    cv2.putText(screen,BloodCellsStr,(35,30),cv2.FONT_HERSHEY_TRIPLEX , 0.5 , (0,0,255),thickness=1)
    cv2.putText(screen, TotalBloodCellsStr, (35, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), thickness=1)
    cv2.imshow("Blood Cell Detector" , Draw_img)
    cv2.imshow("Blood Cell Counter" , screen)

    TotalBloodCellDetection['platelets'] += BloodCellDetection['platelets']
    TotalBloodCellDetection['RBC'] += BloodCellDetection['RBC']
    TotalBloodCellDetection['WBC'] += BloodCellDetection['WBC']

    if cv2.waitKey(1)==ord(' '):break

    print(TotalBloodCellsStr)

cv2.destroyAllWindows()
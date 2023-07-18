import pandas as pd
from torchvision import transforms
import cv2
from utils import make_zip
import torch
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False


def filter_data(predictions, conf):
    sizes = 0
    for score in predictions[2]:
        if score >= conf:
            sizes += 1

    labels = []
    boxes = torch.empty(size=(sizes, len(predictions[1][0])))
    scores = torch.empty(size=(sizes, sizes))

    for i in range(len(predictions[0])):
        point = predictions[2][i]
        if point >= conf:
            labels.append(predictions[0][i])
            boxes[i] = predictions[1][i]
            scores[0][i] = predictions[2][i]

    scores = scores[0]

    return labels, boxes, scores


def draw_image(model, img, conf):
    img = cv2.resize(img, (650, 650), interpolation=cv2.INTER_AREA)

    results = model.predict(img, conf=0.2, iou=0.7)
    names = model.names

    for i, confid in enumerate(results[0].boxes.conf.tolist()):
        if confid >= conf:
            data = results[0].boxes.xyxy[i].tolist()
            label = names[int(results[0].boxes.cls[i])]
            x1, y1, x2, y2 = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            img = cv2.rectangle(img,
                                (x1, y1),
                                (x2, y2),
                                (255, 255, 100), 2)
            img = cv2.putText(img,
                             label, 
                             (x1, y1-10), 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                             (36,255,12), 2)
    
    return img
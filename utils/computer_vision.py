import cv2
import torch
import numpy as np

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


def generate_label_colors(name):
    return np.random.uniform(0, 255, size=(len(name), 3))


def draw_image(model, device, img, conf, colors, time, x_size, y_size):
    results = model.predict(img, device=device, conf=0.2, iou=0.7)
    names = model.names
    parameter = {'label': [],
                 'score': [],
                 'x1': [],
                 'y1': [],
                 'x2': [],
                 'y2': []}
    annotate = {'id': [],
                'x': [],
                'y': [],
                'w': [],
                'h': []}

    for i, confid in enumerate(results[0].boxes.conf.tolist()):
        if confid >= conf:
            data = results[0].boxes.xyxy[i].tolist()
            idx = int(results[0].boxes.cls[i])
            label = names[idx]
            color = colors[int(results[0].boxes.cls[i])]
            x1, y1, x2, y2 = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            x = (x1 + (x2 - x1) / 2) / x_size
            y = (y1 + (y2 - y1) / 2) / y_size
            w = (x2 - x1) / x_size
            h = (y2 - y1) / y_size

            img = cv2.rectangle(img,
                                (x1, y1),
                                (x2, y2),
                                color, 2)
            img = cv2.putText(img,
                              f'LABEL: {label}',
                              (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                              color, 2)
            img = cv2.putText(img,
                              f'ID: {i}',
                              (x1, y1 - 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                              color, 2)

            parameter['label'].append(label)
            parameter['score'].append(conf)
            parameter['x1'].append(x1)
            parameter['y1'].append(y1)
            parameter['x2'].append(x2)
            parameter['y2'].append(y2)

            annotate['id'].append(idx)
            annotate['x'].append(np.round(float(x), decimals=3))
            annotate['y'].append(np.round(float(y), decimals=3))
            annotate['w'].append(np.round(float(w), decimals=3))
            annotate['h'].append(np.round(float(h), decimals=3))

    return img, parameter, annotate

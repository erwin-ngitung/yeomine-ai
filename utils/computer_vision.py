import cv2
import torch
import numpy as np
import av
import pandas as pd
import os

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


def get_time(cap):
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_seconds = frame_count // fps
    seconds = int(total_seconds % 60)
    minutes = int(int(total_seconds / 60) % 60)
    hours = int(int(total_seconds / 3600) % 3600)

    return seconds, minutes, hours


def draw_image(model, device, img, confi, colors, time, x_size, y_size):
    results = model.predict(img, device=device, conf=0.1, iou=0.5)
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
        if confid >= confi:
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
                              cv2.FONT_HERSHEY_DUPLEX, 0.6,
                              color, 2)
            img = cv2.putText(img,
                              f'ID: {idx}',
                              (x1, y1 - 30),
                              cv2.FONT_HERSHEY_DUPLEX, 0.6,
                              color, 2)

            parameter['label'].append(label)
            parameter['score'].append(confid)
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


def recv(frame):
    img = frame.to_ndarray(format="bgr24")
    return av.VideoFrame.from_ndarray(img, format="bgr24")


def count_label(dataset):
    idx = 0
    count = 0
    ids_object = []
    datasets = dataset.sort_values(by=['Label', 'X', 'Y', 'Weight', 'Height']).reset_index(drop=True)

    for label in datasets['Label'].unique():
        count += 1
        ids_object.append(count)
        dataset = datasets[datasets['Label'] == label]

        for i in range(len(dataset)):
            for j in range(1, len(dataset)):

                x1, x2, y1, y2 = dataset['X'].iloc[i], dataset['X'].iloc[j], dataset['Y'].iloc[i], dataset['Y'].iloc[j]
                w1, w2, h1, h2 = \
                    dataset['Weight'].iloc[i], \
                    dataset['Weight'].iloc[j], \
                    dataset['Height'].iloc[i], \
                    dataset['Height'].iloc[j]

                if (abs(x2 - x1) > 0.07 and abs(y1 - y2) > 0.07) or (abs(w2 - w1) > 0.07 and abs(h2 - h1) > 0.07):
                    if i == 0:
                        ids_object.append(count)
                        count += 1
                else:
                    if i == 0:
                        ids_object.append(ids_object[idx + i])
                    else:
                        ids_object[idx + j] = ids_object[idx + i]

        idx += len(dataset)

    datasets['ID'] = ids_object

    replace_value = {}

    for i, index in enumerate(datasets['ID'].unique()):
        replace_value[index] = i

    datasets['ID'].replace(replace_value, inplace=True)

    return datasets


def converter_dataset(path_folder, model):

    dataset = pd.DataFrame(columns=['Label', 'X', 'Y', 'Weight', 'Height'])
    for file in os.listdir(path_folder):
        data = pd.read_fwf(f'{path_folder}/{file}',
                           names=['Label', 'X', 'Y', 'Weight', 'Height'])

        dataset = pd.concat([dataset, data])

    dataset.dropna(inplace=True)
    dataset['Label'] = dataset['Label'].astype(int).replace(model.names)
    dataset['X'] = dataset['X'].astype(float)
    dataset['Y'] = dataset['X'].astype(float)
    dataset['Weight'] = dataset['Weight'].astype(float)
    dataset['Height'] = dataset['Height'].astype(float)
    dataset = dataset.reset_index(drop=True)
    dataset = count_label(dataset)

    return dataset

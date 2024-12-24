import torch
import torch.nn as nn
import math
import os
import cv2
import queue


def seg_split(seg, size=16):
    y, x = seg.shape
    y_num = math.ceil(y / size)
    x_num = math.ceil(x / size)
    
    label_info = torch.zeros(size=(y_num, x_num))
    selected_info = torch.zeros(size=(y_num, x_num))
    for i in range(y_num):
        for j in range(x_num):
            if i == y_num - 1 and j == x_num - 1:
                area = seg[i * size:, j * size:]
            elif i == y_num - 1:
                area = seg[i * size:, j * size: (j + 1) * size]
            elif j == x_num - 1:
                area = seg[i * size: (i + 1) * size, j * size:]
            else:
                area = seg[i * size: (i + 1) * size, j * size: (j + 1) * size]
            if area.max().item() == 4:
                label_info[i, j] = 1

    return label_info, selected_info

# 使用邻域搜索的方法来寻找边界

def select_neighbors(label_info, selected_info, start_point):
    selected_area = [[0, 1], [1, 0], [0, -1], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]
    y, x = start_point
    q = queue.Queue()
    all_area = []
    all_area.append(start_point)
    q.put(start_point)
    selected_info[y, x] = 1
    while not q.empty():
        point = q.get()
        for each in selected_area:
            if point[0] + each[0] >= 0 and point[0] + each[0] < label_info.shape[0] and point[1] + each[1] >= 0 and point[1] + each[1] < label_info.shape[1]:
                if label_info[point[0] + each[0], point[1] + each[1]].item() == 1 and selected_info[point[0] + each[0], point[1] + each[1]].item() == 0:
                    q.put([point[0] + each[0], point[1] + each[1]])
                    all_area.append([point[0] + each[0], point[1] + each[1]])
                    selected_info[point[0] + each[0], point[1] + each[1]] = 1

    return all_area, selected_info


def select_connected_patchs(label_info, selected_info):
    areas = {}
    num = 0
    for i in range(label_info.shape[0]):
        for j in range(label_info.shape[1]):
            if selected_info[i, j].item() == 0 and label_info[i, j].item() == 1:
                selected_area, selected_info = select_neighbors(label_info, selected_info, [i, j])
                areas[num] = selected_area
                num += 1
    return areas


def get_patch_info(path):
    size=16
    seg, _ = torch.load(path)
    seg = seg.squeeze(0)
    image_path = "testDataset/weeddetection/test/test/images"
    image_path = os.path.join(image_path, path.split("/")[-1].replace("_classification.pt", ".png"))
    image = cv2.imread(image_path.replace("_classification.pt", ".png"), cv2.COLOR_BGR2RGB)
    label_info, selected_info = seg_split(seg, size=size)
    areas = select_connected_patchs(label_info, selected_info)
    img_areas = []
    rectangle_list = []
    for each in areas.values():
        each = sorted(each, key=lambda x: x[0])
        y_min = each[0][0]
        y_max = each[-1][0]
        each = sorted(each, key=lambda x: x[1])
        x_min = each[0][1]
        x_max = each[-1][1]
        img_areas.append(image[y_min * size: (y_max + 1) * size, x_min: (x_max + 1) * size, :])
        rectangle_list.append([[x_min * size, y_min * size, (x_max - x_min + 1) * size * 0.9, (y_max - y_min + 1) * size * 0.9]])

    return img_areas, rectangle_list


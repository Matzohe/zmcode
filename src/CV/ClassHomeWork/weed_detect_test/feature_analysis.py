import torch
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
import numpy as np
import os
from tqdm import tqdm

def get_seg_logits_and_labels(path):
    seg_logits, labels = torch.load(path)
    seg_logits = seg_logits.to(dtype=torch.float64)
    seg_logits = maximum_filter(seg_logits, size=(3, 3))
    return seg_logits, labels


# 以圆心为中心，1rating倍半径为左右边界，扩大相关的区域，获得杂草的边界
def get_target_area(seg_logits, labels, rating=1):
    area_info_list = []
    for label in labels:
        x_min = max(0, min(int(label[0][0] - rating * label[1] + 0.5), seg_logits.shape[1] - 1))
        x_max = max(0, min(int(label[0][0] + rating * label[1] + 0.5), seg_logits.shape[1] - 1))
        y_min = max(0, min(int(label[0][1] - rating * label[1] + 0.5), seg_logits.shape[0] - 1))
        y_max = max(0, min(int(label[0][1] + rating * label[1] + 0.5), seg_logits.shape[0] - 1))
        area_info_list.append([[y_min, x_min], [y_max, x_max]])
    return area_info_list

def logits_segmentation(seg_logits, size=16):
    y, x = seg_logits.shape
    if y % size != 0:
        y_number = y // size + 1
    else:
        y_number = y // size
    if x % size != 0:
        x_number = x // size + 1
    else:
        x_number = x // size

    all_info = []

    for i in range(y_number):
        for j in range(x_number):
            if i == y_number - 1 and j == x_number - 1:
                area = seg_logits[i * size:, j * size:]
            elif i == y_number - 1:
                area = seg_logits[i * size:, j * size: (j + 1) * size]
            elif j == x_number - 1:
                area = seg_logits[i * size: (i + 1) * size, j * size:]
            else:
                area = seg_logits[i * size: (i + 1) * size, j * size: (j + 1) * size]
            all_info.append(np.nanmean(area))

    return np.array(all_info)

def analysis_all_seglogits(path):
    seg_logits, _ = get_seg_logits_and_labels(path)
    all_info = logits_segmentation(seg_logits)
    return all_info

def analysis_target_area(path):
    seg_logits, labels = get_seg_logits_and_labels(path)
    area_info_list = get_target_area(seg_logits, labels)
    all_info = []
    for each in area_info_list:
        area_info = seg_logits[each[0][1]:each[1][1], each[0][0]:each[1][0]]
        area_info = area_info.reshape(-1)
        all_info.append(np.nanmean(area_info))
    
    # all_info = np.concatenate(all_info, axis=0)
    all_info = np.array(all_info)
    
    return all_info

def label_area_distribution(path):
    seg_logits, labels = get_seg_logits_and_labels(path)
    labels = get_target_area(seg_logits, labels)
    area_info_list = []
    for each in labels:
        area_info_list.append(((each[1][0] - each[0][0]) * (each[1][1] - each[0][1])))
    return np.array(area_info_list)


def main():
    path_list = [os.path.join("experiment/output/weed_detect/seg_logits/", each) for each in os.listdir("experiment/output/weed_detect/seg_logits/")]
    final_info = []

    for each in tqdm(path_list, total=len(path_list)):
        all_info = label_area_distribution(each)
        try:
            final_info.append(all_info.reshape(-1))
        except:
            print(each, all_info)
            continue
    final_info = np.concatenate(final_info, axis=-1)
    final_info.astype(np.float64)
    print(np.nansum(final_info) / len(final_info))
    plt.hist(final_info, bins=256, range=(0, max(final_info)), color='green', alpha=0.7)
    plt.show()
    plt.close()

    final_info = []
    for each in tqdm(path_list, total=len(path_list)):
        all_info = analysis_target_area(each)

        final_info.append(all_info.reshape(-1))
    final_info = np.concatenate(final_info, axis=-1)
    final_info.astype(np.float64)
    print(np.nanmedian(final_info))
    plt.hist(final_info, bins=256, range=(0, 1), color='green', alpha=0.7)
    plt.show()
    plt.close()
    final_info = []
    for each in tqdm(path_list, total=len(path_list)):
        all_info = analysis_all_seglogits(each)

        try:
            final_info.append(all_info.reshape(-1))
        except:
            print(each, all_info)
            continue

    final_info = np.concatenate(final_info, axis=0)
    final_info.astype(np.float64)
    print(np.median(final_info))
    plt.hist(final_info, bins=256, range=(0, 1), color='green', alpha=0.7)
    plt.show()

        

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image


def NACLIP_Drawer(data_samples, specific_target_index=None, 
                  save=False, seg_save_path=None, heatmap_save_path=None):
    """_summary_

    Args:
        data_samples (List[Dict]): the output of NACLIP, is a List contains n pixture dict, 
                                each dict contain two keys: seg_logits and pred_sem_seg

                                seg_logits: each label's softmax posibility, it's shape is (number of classes, height, width)
                                pred_sem_seg: the predict label, it's shape is (height, width), the argmax result of seg_logits
        
        specific_target_index (int, optional): Use to generate a posibility map for a specific target index. Defaults to None.

        save (bool): Use to save the image. Defaults to False.

        seg_save_path (str, optional): Save path. Defaults to None.

        heatmap_save_path (str, optional): heatmap Save path, for sementic segmentation, restrict not None specific_target_index. Defaults to None.


    Returns:
        SegmentedImage: List of Segmented Images

        PosibilityHeatMap (optional): List of Posibility HeatMap or None

    """

    default_color_list = [[0, 0, 0], [0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192], [0, 64, 64], [0, 192, 224],
                 [0, 192, 192], [128, 192, 64], [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224], [0, 0, 64],
                 [0, 160, 192], [128, 0, 96], [128, 0, 192], [0, 32, 192], [128, 128, 224], [0, 0, 192],
                 [128, 160, 192],
                 [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128], [64, 128, 32], [0, 160, 0], [0, 0, 0],
                 [192, 128, 160], [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0], [0, 128, 0], [192, 128, 32],
                 [128, 96, 128], [0, 0, 128], [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160], [0, 96, 128],
                 [128, 128, 128], [64, 0, 160], [128, 224, 128], [128, 128, 64], [192, 0, 32],
                 [128, 96, 0], [128, 0, 192], [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160], [64, 96, 0],
                 [0, 128, 192], [0, 128, 160], [192, 224, 0], [0, 128, 64], [128, 128, 32], [192, 32, 128],
                 [0, 64, 192],
                 [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160], [64, 32, 128], [128, 192, 192], [0, 0, 160],
                 [192, 160, 128], [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128], [64, 128, 96],
                 [64, 160, 0],
                 [0, 64, 0], [192, 128, 224], [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0]]

    fig_number = len(data_samples)

    if save:
        assert seg_save_path is not None, "seg save path should not be None when save is True"
        assert len(seg_save_path) == fig_number, "seg save path's length should be equal to fig number"

        if heatmap_save_path is not None:
            assert heatmap_save_path is not None, "heatmap save path should not be None when save is True and heatmap_save_path is not None"
            assert len(heatmap_save_path) == fig_number, "heatmap save path's length should be equal to fig number"

    for i in range(fig_number):
        pred_sem_seg = data_samples[i]['pred_sem_seg'][0]
        seg_logits = data_samples[i]['seg_logits']

        output_image = torch.zeros(size=(pred_sem_seg.shape[-2], pred_sem_seg.shape[-2], 3))
        for j in tqdm(range(pred_sem_seg.shape[-2]), total=pred_sem_seg.shape[-1], desc="processing Image {}".format(i)):
            for k in range(pred_sem_seg.shape[1]):
                output_image[j, k] = torch.tensor(default_color_list[pred_sem_seg[j, k]])
        
        image = output_image.numpy().astype(np.uint8)
        if save:
            save_root = seg_save_path[i]
            image = Image.fromarray(image)
            image.save(save_root)
        else:
            plt.figure(figsize=(12, 12))
            plt.imshow(image)
            plt.show()
            plt.close()
        
        if specific_target_index is not None:
            target_img = seg_logits[i].numpy()
            plt.imshow(target_img[1], cmap="inferno")
            plt.colorbar()
            if save:
                save_root = heatmap_save_path[i]
                plt.savefig(save_root)
                plt.close()
            else:
                plt.show()
                plt.close()

            
        



        


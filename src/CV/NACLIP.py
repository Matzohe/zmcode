from ..MultiModal import NACLIP as clip
from ..utils.pamr import PAMR
from ..utils.NACLIPPrompts.imagenet_template import openai_imagenet_template
import logging
from mmengine.structures import PixelData
from typing import List, Union, Tuple
import torch
import torch.nn as nn


class NACLIP(nn.Module):
    def __init__(self, config, labels: Union[List[str], Tuple[str], str, None] = None):
        super().__init__()
        self.config = config
        self.device = config.INFERENCE['device']
        self.label_number = None
        self.dtype = None  # the same with the query encoding
        self.model, _ = clip.load(name=config.NACLIP['model_name'], device=self.device)
        self.model.visual.set_params(arch=config.NACLIP['arch'], 
                                     attn_strategy=config.NACLIP['attn_strategy'], 
                                     gaussian_std=float(config.NACLIP['gaussian_std']))
        self.pamr = PAMR(num_iter=int(config.NACLIP['pamr_steps']), dilations=eval(config.NACLIP['pamr_stride']))
        self.logit_scale = int(config.NACLIP['logit_scale'])
        self.prob_thd = float(config.NACLIP['prob_thd'])
        self.slide_stride = int(config.NACLIP['slide_stride'])
        self.slide_crop = int(config.NACLIP['slide_crop'])
        self.align_corners = False
        self.Label2Sequence(labels)

        # use to segment the specific colors, if the number of labels is bigger than the size of default color list, we use % to process the label color
        self.default_color_list = [[0, 0, 0], [0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192], [0, 64, 64], [0, 192, 224],
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

        logging.info('attn_strategy is {}, arch is {} & Gaussian std is {}'.format(config.NACLIP['attn_strategy'], 
                                                                                   config.NACLIP['arch'], 
                                                                                   config.NACLIP['gaussian_std']))

    def Label2Sequence(self, labels: Union[List[str], Tuple[str], str, None]):
        """_summary_

        Args:
            labels (Union[List[str], str]): The sementics that used to segment the image

        Raises:
            ValueError: labels should be a str or list or tuple of str

        Returns:
            _type_: a tensor that contains the sementics, each sementics has been changed into several
                    sentences follows the guides in the openai_imagenet_template(it's a list) and take
                    the average embeddings
        """


        if labels is None:
            labels = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'fish', 'cat', 'dog', 'horse',
                 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        if isinstance(labels, str):
            labels = [labels]

        query_list = []
        with torch.no_grad():
            for query in labels:
                guides = clip.tokenize([mode(query) for mode in openai_imagenet_template]).to(device=self.device)
                guides = self.model.encode_text(guides)
                guides /= guides.norm(dim=-1, keepdim=True)
                guides = guides.mean(dim=0)
                guides /= guides.norm(dim=0)
                query_list.append(guides.unsqueeze(0))
        self.query_feature = torch.cat(query_list, dim=0)
        self.dtype = self.query_feature.dtype
        self.label_number = len(self.query_feature)
        self.query_idx = [i for i in range(len(self.query_feature))]
        return self.query_feature
    
    def forward_feature(self, img):
        """_summary_

        Args:
            img (_type_): excepted to be a tensor, which shape is [batch_size, channel_number, width, hight]

        Returns:
            _type_: the logits of each pixel, which shape is [batch_size, label_number, width, hight], label number equals the embedding labels
        """


        if type(img) == list:
            img = img[0]

        image_features = self.model.encode_image(img, return_all=True)
        image_features = image_features[:, 1:]
        image_features /= image_features.norm(dim=-1, keepdim=True)

        logits = image_features @ self.query_feature.T

        patch_size = self.model.visual.patch_size
        w, h = img[0].shape[-2] // patch_size, img[0].shape[-1] // patch_size
        out_dim = logits.shape[-1]
        logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h)
        logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear', align_corners=self.align_corners)
        return logits

    def forward_slide(self, img, stride=112, crop_size=224):
        """
        Inference by sliding-window with overlap. If h_crop > h_img or w_crop > w_img,
        the small patch will be used to decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.label_number
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.forward_feature(crop_img)
                preds += nn.functional.pad(crop_seg_logit,
                                           (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        logits = preds / count_mat
        return logits

    def predict(self, inputs, data_samples=None):
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            batch_img_metas = [dict(
                ori_shape=inputs.shape[2:],
                img_shape=inputs.shape[2:],
                pad_shape=inputs.shape[2:],
                padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        if self.slide_crop > 0:
            seg_logits = self.forward_slide(inputs, self.slide_stride, self.slide_crop)
        else:
            seg_logits = self.forward_feature(inputs)

        img_size = batch_img_metas[0]['ori_shape']
        seg_logits = nn.functional.interpolate(seg_logits, size=img_size, mode='bilinear', align_corners=self.align_corners)

        if self.pamr:
            img = nn.functional.interpolate(inputs, size=img_size, mode='bilinear', align_corners=self.align_corners)
            try:
                # This step will consume a huge number of Memory when processing large Image, so here we use cpu
                new_img = img.to(device="cpu")
                original_dtype = seg_logits.dtype
                new_seg_logits = seg_logits.to(device="cpu")
                seg_logits = self.pamr(new_img, new_seg_logits.to(img.dtype)).to(original_dtype)
                seg_logits.to(device=self.device)
            except RuntimeError as e:
                logging.warning(f"Couldn't apply PAMR for image {batch_img_metas[0]['img_path'].split('/')[-1]} "
                                f"of size {img_size}, probably due to low memory. Error message: \"{str(e)}\"")

        return self.postprocess_result(seg_logits, data_samples)

    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        if data_samples is None:
            data_samples = [{} for _ in range(batch_size)]
        for i in range(batch_size):
            seg_probs = torch.softmax(seg_logits[i] * self.logit_scale, dim=0)  # n_queries * w * h

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_probs = seg_probs.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_probs = (seg_probs * cls_index).max(1)[0]

            seg_pred = seg_probs.argmax(0, keepdim=True)
            seg_pred[seg_probs.max(0, keepdim=True)[0] < self.prob_thd] = 0
            seg_probs /= seg_probs.sum(0, keepdim=True)


            data_samples[i]['seg_logits'] = seg_probs
            data_samples[i]['pred_sem_seg'] = seg_pred
        
        return data_samples

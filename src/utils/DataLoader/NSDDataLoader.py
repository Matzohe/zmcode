import torch
import numpy as np
import pickle
import nibabel as nib
from typing import List, Optional, Union
from .utils.NSD_zscore import zscore_by_run


class NSDDataset:
    def __init__(self, config):
        # Loading Config from config path or dataclass
        
        # list the parameters that have been used in the code
        # all the parameters is comming from the config dataclass
        self.repo = int(config.NSD['repo'])
        self.stimuli_info = config.NSD['stimuli_info']
        self.general_mask_root = config.NSD['general_mask_root']
        self.roi_mask_root = config.NSD['roi_mask_root']
        self.session_num = int(config.NSD['session_num'])
        self.beta_value_root = config.NSD['beta_value_root']

        self.image_index_save_path = config.NSD['image_index_save_path']
        self.image_trail_save_path = config.NSD['image_trail_save_path']
        self.roi_mask_save_root = config.NSD['roi_mask_save_root']
        self.general_mask_save_root = config.NSD['general_mask_save_root']
        self.voxal_response_save_root = config.NSD['voxal_response_save_root']
        self.nonzero_mask_save_root = config.NSD['nonzero_mask_save_root']
    
    # Get the NSD dataset stimulate infomation, the image index for each subject
    def extract_image_index(self, 
                            subj: int, 
                            save = False) -> List[int]:

        stim_info = pickle.load(self.stimuli_info)
        key = "subject{}_rep0".format(subj)
        image_index_list = stim_info.cocoId[stim_info[key] != 0]

        if save:
            save_path = self.image_index_save_path.format(subj)
            torch.save(image_index_list, save_path)

        return image_index_list
    
    # Get the NSD dataset stimulate infomation, the trail index (10000 * 3) for each subject, each row is in commone with the image index list
    def extract_trail_index(self, 
                            subj: int, 
                            save = False) -> List[List[int]]:
        # Get the fmri signal index for each image in test for a certain subj
        # each image has 3 trails, finally we will get 10000 * 3 trails

        stim_info = pickle.load(self.stimuli_info)
        key = "subject{}_rep{}"
        trail_index = []
        for i in range(self.repo):
            select_info = key.format(subj, i)
            trail_index.append(stim_info[select_info][stim_info[select_info]] != 0)
        trail_index = np.array(trail_index).T - 1  # Turn to shape [10000, 3], and change from 1 based to 0 based

        if save:
            save_path = self.image_trail_save_path.format(subj)
            torch.save(trail_index, save_path)

        return trail_index

    # Get the ROI mask for a certain subject, if roi is none, get the general mask for the subject
    def extract_roi_mask(self, 
                         subj: int, 
                         roi_name: str = "",
                         save=False) -> np.ndarray:
        # Get the ROI mask for a certain subject
        if roi_name == "": # cortical
            general_mask_root = self.general_mask_root
            general_mask_root = general_mask_root.format(subj)
            general_mask = nib.load(general_mask_root).get_fdata()
            general_mask = general_mask > -1

            if save:
                save_root = self.general_mask_save_root.format(subj, roi_name)
                torch.save(general_mask, save_root)

            return general_mask
        
        else:
            general_mask_root = self.general_mask_root.format(subj)
            general_mask = nib.load(general_mask_root).get_fdata()

            roi_mask_root = self.roi_mask_root.format(subj, roi_name)
            roi_mask = nib.load(roi_mask_root).get_fdata()

            new_roi_mask = roi_mask > 0
            cortical = general_mask > -1

            roi_1d_mask = roi_mask[cortical].astype(int)

            if save:
                save_root = self.roi_mask_save_root.format(subj, roi_name)
                cortical_root = self.general_mask_save_root.format(subj, roi_name)
                torch.save(roi_1d_mask, save_root)
                torch.save(new_roi_mask, cortical_root)

            return new_roi_mask

    # Get the voxal response for a certain subject, get voxal responses for the subj and the target roi, for all 40 sessions
    # meanwhile, choose to process zscore process
    def extract_voxal_activation(self, 
                                 subj: int, 
                                 roi_name: str = "",
                                 save=True,
                                 zscore=True
                                 ):
        try:
            mask = torch.load(self.general_mask_save_root.format(subj, roi_name))
        except:
            mask = self.extract_roi_mask(subj=subj, roi_name=roi_name, save=True)
        
        # for each person, there is 40 session, and each session has 750 figs
        # We will target subject's 30000 fMRI activation
        # The output will process zscore by default
        
        cortical_beta_mat = None

        for session in range(self.session_num):
            session_num = session + 1
            beta_value_root = self.beta_value_root.format(subj, session_num)

            fmri_data = nib.load(beta_value_root)
            
            beta = fmri_data.get_fdata()
            cortical_beta = (beta[mask]).T  # verify the mask with array

            if cortical_beta_mat is None:
                cortical_beta_mat = cortical_beta / 300
            
            else:
                cortical_beta_mat = np.vstack((cortical_beta_mat, cortical_beta / 300))

        if zscore:
            print("Zscoring...")
            cortical_beta_mat = zscore_by_run(cortical_beta_mat)
            finite_flag = np.all(np.isfinite(cortical_beta_mat))
            print("Is finite:" + str(finite_flag))

            if finite_flag == False:
                nonzero_mask = (
                    np.sum(np.isfinite(cortical_beta_mat), axis=0)
                    == cortical_beta_mat.shape[0]
                )
                nonzero_mask_save_root = self.nonzero_mask_save_root.format(subj, roi_name)
                torch.save(nonzero_mask, nonzero_mask_save_root)
            
        if save:
            save_root = self.voxal_response_save_root.format(subj)
            torch.save(cortical_beta_mat, save_root)

        return cortical_beta_mat

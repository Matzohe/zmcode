import torch
import numpy as np
import pickle
import nibabel as nib
from typing import List, Optional, Union
from tqdm import tqdm
import pandas as pd
from .utils.NSD_utils import zscore_by_run, ev
from ..utils import check_path, check_tensor


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

        self.image_index_save_root = config.NSD['image_index_save_root']
        self.image_root_save_root = config.NSD['image_root_save_root']
        self.image_trail_save_path = config.NSD['image_trail_save_root']
        self.roi_mask_save_root = config.NSD['roi_mask_save_root']
        self.general_mask_save_root = config.NSD['general_mask_save_root']
        self.voxal_zscore_response_save_root = config.NSD['voxal_zscore_response_save_root']
        self.voxal_nonzscore_response_save_root = config.NSD['voxal_nonzscore_response_save_root']
        self.nonzero_mask_save_root = config.NSD['nonzero_mask_save_root']
        self.zscore_avg_activation_save_root = config.NSD['zscore_avg_activation_save_root']
        self.zscore_activation_ev_save_root = config.NSD['zscore_activation_ev_save_root']
        self.nonzscore_avg_activation_save_root = config.NSD['nonzscore_avg_activation_save_root']
        self.nonzscore_activation_ev_save_root = config.NSD['nonzscore_activation_ev_save_root']

        self.pure_response_save_root = config.NSD['pure_response_save_root']
    
    # Get the NSD dataset stimulate infomation, the image index for each subject
    def extract_image_index(self, 
                            subj: int, 
                            save = False, 
                            ) -> List[int]:

        stim_info = pd.read_pickle(self.stimuli_info)
        key = "subject{}_rep0".format(subj)
        image_index_list = list(stim_info.cocoId[stim_info[key] != 0])

        if save:
            save_path = self.image_index_save_root.format(subj)
            check_path(save_path)
            torch.save(image_index_list, save_path)

        return image_index_list
    
    def extract_image_root(self,
                           subj: int,
                           save = False,
                           ) -> List[str]:
        
        stim_info = pd.read_pickle(self.stimuli_info)
        key = "subject{}_rep0".format(subj)
        image_root_list = list(stim_info.cocoSplit[stim_info[key] != 0])
        image_index_list = list(stim_info.cocoId[stim_info[key] != 0])
        image_root_list = ["/".join([i, "{:012}.jpg".format(j)]) for i, j in zip(image_root_list, image_index_list)]
        if save:
            save_path = self.image_root_save_root.format(subj)
            check_path(save_path)
            torch.save(image_root_list, save_path)
        
        return image_root_list


    # Get the NSD dataset stimulate infomation, the trail index (10000 * 3) for each subject, each row is in commone with the image index list
    def extract_trail_index(self, 
                            subj: int, 
                            save = False, 
                            ) -> np.ndarray:
        # Get the fmri signal index for each image in test for a certain subj
        # each image has 3 trails, finally we will get 10000 * 3 trails

        stim_info = pd.read_pickle(self.stimuli_info)
        key = "subject%d_rep%01d"
        trail_index = []
        for i in range(self.repo):
            select_info = key % (subj, i)
            trail_index.append(list(stim_info[select_info][stim_info[select_info] != 0]))
        trail_index = np.array(trail_index).T - 1  # Turn to shape [10000, 3], and change from 1 based to 0 based

        if save:
            save_path = self.image_trail_save_path.format(subj)
            check_path(save_path)
            torch.save(trail_index, save_path)

        return trail_index

    # Get the ROI mask for a certain subject, if roi is none, get the general mask for the subject
    def extract_roi_mask(self, 
                         subj: int, 
                         roi_name: str = "",
                         save=False
                         ) -> np.ndarray:
        # Get the ROI mask for a certain subject
        if roi_name == "": # cortical
            general_mask_root = self.general_mask_root
            general_mask_root = general_mask_root.format(subj)
            general_mask = nib.load(general_mask_root).get_fdata()
            general_mask = general_mask > -1

            if save:
                save_path = self.general_mask_save_root.format(subj, roi_name)
                check_path(save_path)
                torch.save(general_mask, save_path)

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
                save_path = self.roi_mask_save_root.format(subj, roi_name)
                cortical_path = self.general_mask_save_root.format(subj, roi_name)
                check_path(save_path)
                check_path(cortical_path)
                torch.save(roi_1d_mask, save_path)
                torch.save(new_roi_mask, cortical_path)

            return new_roi_mask

    # Get the voxal response for a certain subject, get voxal responses for the subj and the target roi, for all 40 sessions
    # meanwhile, choose to process zscore process
    def extract_voxal_activation(self, 
                                 subj: int, 
                                 roi_name: str = "",
                                 save=True,
                                 zscore=False, 
                                 ):
        try:
            mask = torch.load(self.general_mask_save_root.format(subj, roi_name))
        except:
            mask = self.extract_roi_mask(subj=subj, roi_name=roi_name, save=True)
        
        mask = check_tensor(mask)

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
            if zscore:
                save_path = self.voxal_zscore_response_save_root.format(subj, roi_name)
            else:
                save_path = self.voxal_nonzscore_response_save_root.format(subj, roi_name)
            check_path(save_path)
            torch.save(cortical_beta_mat, save_path)

        return cortical_beta_mat


# compute an average voxal activation for three time repeate images
    def compute_ev(self,
                   subj: int,
                   roi_name: str = "",
                   zscored: bool = False, 
                   biascorr: bool = False,
                   save: bool = True,
                   ) -> torch.Tensor:
        trail_root = self.image_trail_save_path.format(subj)
        if zscored:
            response_root = self.voxal_zscore_response_save_root.format(subj, roi_name)
        else:
            response_root = self.voxal_nonzscore_response_save_root.format(subj, roi_name)

        response_data = torch.load(response_root)
        trail_data = torch.load(trail_root)
        repeat_num = trail_data.shape[0]

        ev_list = []
        average_voxal_activation = torch.zeros(size=(repeat_num, response_data.shape[1]))

        for v in tqdm(range(response_data.shape[1]), desc="computing everage voxal activations"):
            repo_data_list = []
            for r in range(self.repo):
                repo_data_list.append(response_data[trail_data[:, r], v])
            repo_data_list = torch.tensor(repo_data_list).T

            average_voxal_activation[:, v] = torch.mean(repo_data_list, dim=1)

            ev_list.append(ev(repo_data_list, biascorr=biascorr))

        ev_list = torch.tensor(ev_list)

        if save:
            if zscored:
                avg_avtivation_save_path = self.zscore_avg_activation_save_root.format(subj, roi_name)
                ev_avtivation_save_path = self.zscore_activation_ev_save_root.format(subj, roi_name)
            else:
                avg_avtivation_save_path = self.nonzscore_avg_activation_save_root.format(subj, roi_name)
                ev_avtivation_save_path = self.nonzscore_activation_ev_save_root.format(subj, roi_name)
            check_path(avg_avtivation_save_path)
            check_path(ev_avtivation_save_path)
            torch.save(average_voxal_activation, avg_avtivation_save_path)
            torch.save(ev_list, ev_avtivation_save_path)
        
        return ev_list
        
    def get_pure_activation(self,
                            subj: int,
                            roi_name: str = "",
                            zscored: bool = False,
                            save: bool = True
                            ) -> torch.Tensor:
        
        trail_root = self.image_trail_save_path.format(subj)

        if zscored:
            response_root = self.voxal_zscore_response_save_root.format(subj, roi_name)
        else:
            response_root = self.voxal_nonzscore_response_save_root.format(subj, roi_name)

        response_data = torch.load(response_root)
        trail_data = torch.load(trail_root)

        pure_list = []

        for v in tqdm(range(response_data.shape[1]), desc="computing everage voxal activations"):
            repo_data_list = []
            for r in range(self.repo):
                repo_data_list.append(response_data[trail_data[:, r], v])
            
            repo_data_list = torch.tensor(repo_data_list).T.unsqueeze(1) # 10000 * 1 * 3
            
            pure_list.append(repo_data_list)
            
        pure_list = torch.cat(pure_list, dim=1) # 10000 * voxal_num * 3

        if save:
            pure_response_save_path = self.pure_response_save_root.format(subj, roi_name)
            check_path(pure_response_save_path)
            torch.save(pure_list, pure_response_save_path)

        return pure_list
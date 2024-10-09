from src.utils.utils import INIconfig
from src.utils.DataLoader.NSDDataLoader import NSDDataset
from src.utils.NSD_CKA import CKAforNSD


def test_nsd_dataset():
    config = INIconfig("config.cfg")
    dataset = NSDDataset(config)

    dataset.extract_image_index(subj=1, save=True)
    dataset.extract_trail_index(subj=1, save=True)

    dataset.extract_roi_mask(subj=1, roi_name="SELECTIVE_ROI", save=True)
    dataset.extract_voxal_activation(subj=1, roi_name="SELECTIVE_ROI", zscore=True, save=True)
    dataset.extract_voxal_activation(subj=1, roi_name="SELECTIVE_ROI", zscore=False, save=True)

    dataset.compute_ev(subj=1, roi_name="SELECTIVE_ROI", zscored=True, biascorr=False, save=True)
    dataset.compute_ev(subj=1, roi_name="SELECTIVE_ROI", zscored=False, biascorr=False, save=True)

    dataset.get_pure_activation(subj=1, roi_name="SELECTIVE_ROI", zscored=False, save=True)

    cka = CKAforNSD(config)
    cka.process(subj1=1, roi1="SELECTIVE_ROI", save=True)

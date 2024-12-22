from src.Recognize.BrainDiVE.BrainDiVE import BrainDiVE
from src.utils.utils import INIconfig
from src.Recognize.BrainDiVE.BrainDiVE_utils.diffusion import ImageOutput
from src.utils.utils import set_seed
import os
import torch

def BrainDiVETest():
    config = INIconfig('config.cfg')
    brain_area = ["FFA", "EBA", "RSC", "VWFA", "FOOD"]
    model = BrainDiVE(config).to(device=config.INFERENCE['device'])
    model.load_weight(config.BRAINDIVE["weight_root"])
    seed_list = torch.randint(0, 100000, size=(1000, ))
    for subj_id in range(1, 2):
        for j, roi in enumerate(brain_area):
            model.load_roi(config.BRAINDIVE['roi_root'].format(roi))
            for i in range(int(config.BRAINDIVE['image_number'])):
                set_seed(seed_list[j * 100 + i])
                latent = model(subj_id=subj_id, roi=roi, save_middle_image=False, save_middle_grad=False, fig_id=i+1)
                image = model.pipe.ImageDecoding(latent)
                image = ImageOutput(image)
                image_save_root = config.BRAINDIVE['final_image_save_root'].format(subj_id, roi, i+1)
                if not os.path.exists('/'.join(image_save_root.split('/')[:-1])):
                    os.makedirs('/'.join(image_save_root.split('/')[:-1]))
                image.save(image_save_root)





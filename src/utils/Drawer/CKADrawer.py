import numpy as np
import os
import matplotlib.pyplot as plt



def CKA_Image_Saver(cka_result, config):
    cka_matrix = np.flipud(cka_result['CKA'].numpy())
    model1_name = cka_result['model1_name']
    model2_name = cka_result['model2_name']
    save_root = config.CKA['image_save_root']
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    save_name = config.CKA['image_save_name'].format(model1_name, model2_name)
    save_file = os.path.join(save_root, save_name)
    dpi = int(config.CKA['dpi'])
    figsize = eval(config.CKA['figsize'])
    plt.figure(figsize=figsize)
    plt.imshow(cka_matrix, cmap="inferno")
    plt.colorbar()
    plt.savefig(save_file, dpi=dpi)
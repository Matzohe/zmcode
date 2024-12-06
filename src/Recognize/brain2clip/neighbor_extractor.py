# extract the target point's eight neighbors or twenty six neighbors
# starting from the central point
import torch
import numpy as np
import queue
import random

def extract_neighbors(mask: np.ndarray,
                      is_find_list: torch.tensor, 
                      target_point: torch.tensor,
                      is_eight_neighbors: bool=False) -> queue.Queue:
    # note: even these point's name is x, y, z, it dosen't mean actually x axis, y axis and z axis
    # it's just mean the first, second and the third dim of the data
    x, y, z = target_point
    Q = queue.Queue()
    x_list = [0, -1, 1]
    y_list = [0, -1, 1]
    z_list = [0, -1, 1]
    random.shuffle(x_list)
    random.shuffle(y_list)
    random.shuffle(z_list)
    for i in x_list:
        for j in y_list:
            for k in z_list:
                _x = max(0, min(mask.shape[0] - 1, x + i))
                _y = max(0, min(mask.shape[1] - 1, y + j))
                _z = max(0, min(mask.shape[2] - 1, z + k))
                
                if _x == x and _y == y and _z == z:
                    continue

                if is_eight_neighbors:
                    if abs(i) + abs(j) + abs(k) > 1:
                        continue
                if mask[_x, _y, _z] > 0 and not is_find_list[_x, _y, _z]:
                    Q.put((_x, _y, _z))
                    is_find_list[_x, _y, _z] = True
    return Q

def seed_point_search(*, 
                          mask,
                          seed_point,
                          target_model_embedding, 
                          fMRI_activation,
                          coodinate2index, 
                          is_eight_neighbors=False, 
                          ) -> torch.tensor:
    current_activation = torch.zeros_like(target_model_embedding)
    current_activation += fMRI_activation[coodinate2index[seed_point]]

    voxel_selected_time_list = torch.zeros(size=(fMRI_activation.shape[0], ))
    voxel_selected_time_list[coodinate2index[seed_point]] = 1
    is_find_list = torch.zeros(size=mask.shape).to(dtype=torch.bool)
    is_find_list[seed_point[0], seed_point[1], seed_point[2]] = True

    serch_Q = queue.Queue()
    serch_Q.put(seed_point)
    
    while not serch_Q.empty():
        target_point = serch_Q.get()
        neighbor_Q = extract_neighbors(mask, is_find_list, target_point, is_eight_neighbors)

        while not neighbor_Q.empty():
            neighbor_point = neighbor_Q.get()
            new_activation = current_activation + fMRI_activation[coodinate2index[neighbor_point]]
            if torch.norm(new_activation - target_model_embedding) < torch.norm(current_activation - target_model_embedding):
                current_activation = new_activation
                serch_Q.put(neighbor_point)
                voxel_selected_time_list[coodinate2index[neighbor_point]] = 1

    return voxel_selected_time_list
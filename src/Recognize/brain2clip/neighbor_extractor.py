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
    """_summary_

    Args:
        mask (np.ndarray): dim=3, the mask of the brain, if mask[i, j, k] > 0, the point(x, y, z) is in the brain and useful.
        is_find_list (torch.tensor, dtype=torch.bool): dim=3, is_find_list[i, j, k] = True means the point have been checked. 
        target_point (torch.tensor): the target point to extract the 26 neighbor.
        is_eight_neighbors (bool, optional): Defaults to False, extracte the 8 neighbor/26 neighbor.

    Returns:
        queue.Queue: a queue contains the target point's valid 26 or 8 neighbors
    """
    # note: even these point's name is x, y, z, it dosen't mean actually x axis, y axis and z axis
    # it's just mean the first, second and the third dim of the data
    x, y, z = target_point
    Q = queue.Queue()
    # get the neighbor's coodinate, to avoid the affect of the order of selecting the neighbors,
    # shuffle the list before extract the neighbors
    x_list = [0, -1, 1]
    y_list = [0, -1, 1]
    z_list = [0, -1, 1]
    random.shuffle(x_list)
    random.shuffle(y_list)
    random.shuffle(z_list)
    for i in x_list:
        for j in y_list:
            for k in z_list:
                # check if the point is within the mask
                _x = max(0, min(mask.shape[0] - 1, x + i))
                _y = max(0, min(mask.shape[1] - 1, y + j))
                _z = max(0, min(mask.shape[2] - 1, z + k))
                # if the point equals the target point, continue
                if _x == x and _y == y and _z == z:
                    continue
                # check if the point is in 8 neighbor, only work when is_eight_neighbors is True
                if is_eight_neighbors:
                    if abs(i) + abs(j) + abs(k) > 1:
                        continue
                # check if the point is a valid neighbor point that mask[_x, _y, _z] > 0 and not been found yet
                if mask[_x, _y, _z] > 0 and not is_find_list[_x, _y, _z]:
                    Q.put((_x, _y, _z))
                    is_find_list[_x, _y, _z] = True
    return Q

def seed_point_search(*, 
                      mask,
                      seed_point,
                      target_model_embedding, 
                      fMRI_activation,coodinate2index, 
                      is_eight_neighbors=False, 
                      dtype=torch.float64
                      ) -> torch.tensor:
    """_summary_

    Args:
        mask (_type_): dim=3, the mask of the brain, if mask[i, j, k] > 0, the point(x, y, z) is in the brain and useful.
        seed_point (_type_): the start point of the bfs, a tuple of (x0, y0, z0)
        target_model_embedding (torch.tensor): the optimization target, shape: (1, embedding_dim)
        fMRI_activation (torch.tensor): 2D activation, shape:(voxel_num), means each voxel's activation vector.
                                        the goal is make the sum of several voxel's activation vector close to the target_model_embedding

        coodinate2index (dict): a dict which key is a tuple of (x, y, z) and value is the voxel index in fMRI_activation.
                                as the fMRI_activation has been flattened, it's used to find out the original coodinate of a voxel
                                for example, coodinate2index[(x, y, z)] = i, means voxel i's coodinate is (x, y, z) 
                                and voxel i's activation vector is fMRI_activation[i]

        is_eight_neighbors (bool, optional): Defaults to False, extracte the 8 neighbor/26 neighbor.
        dtype (_type_, optional): Defaults to torch.float64, the data type

    Returns:
        torch.tensor: a list that save the information of which voxel has been selec
    """
    current_activation = torch.zeros_like(target_model_embedding).to(dtype=dtype)
    current_activation += fMRI_activation[coodinate2index[seed_point]].to(dtype=dtype)

    voxel_selected_time_list = torch.zeros(size=(fMRI_activation.shape[0], ))
    voxel_selected_time_list[coodinate2index[seed_point]] = 1
    is_find_list = torch.zeros(size=mask.shape).to(dtype=torch.bool)
    is_find_list[seed_point[0], seed_point[1], seed_point[2]] = True

    target_model_embedding = target_model_embedding.to(dtype=dtype)
    serch_Q = queue.Queue()
    serch_Q.put(seed_point)
    
    while not serch_Q.empty():
        target_point = serch_Q.get()
        neighbor_Q = extract_neighbors(mask, is_find_list, target_point, is_eight_neighbors)

        while not neighbor_Q.empty():
            neighbor_point = neighbor_Q.get()
            new_activation = current_activation + fMRI_activation[coodinate2index[neighbor_point]]
            if (new_activation * target_model_embedding).sum() > 0 and torch.norm(new_activation - target_model_embedding) < torch.norm(current_activation - target_model_embedding):
                current_activation = new_activation
                serch_Q.put(neighbor_point)
                voxel_selected_time_list[coodinate2index[neighbor_point]] = 1
    return voxel_selected_time_list
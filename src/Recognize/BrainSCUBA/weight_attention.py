import torch
import torch.nn.functional as F

def WeightTransform(image_embedding, voxel_weight, temperature=1/150):
    """_summary_

    Args:
        image_embedding (torch.tensor): the image embedding of clip
        voxel_weight (torch.tensor): the ridge regression or linear weight of clip2brain

    returns:
        adapted_weight: the adapted weight
    """

    assert image_embedding.shape[1] == voxel_weight.shape[1] and voxel_weight.ndim == 2, "the shape of image_embedding and voxel_weight is not correct"

    middle_matrix = voxel_weight @ image_embedding.T / temperature
    score = torch.softmax(middle_matrix, dim=1)

    # compute the L2 norm of the image embedding, and the image embedding divided by the L2 norm
    avg_embedding = torch.norm(image_embedding, dim=1, p=2).view(image_embedding.shape[0], 1)
    direct_vector = image_embedding / avg_embedding

    output = (score @ avg_embedding) * (score @ direct_vector)

    return output





def cosine_similarity(x, y):

    cosine_similarity_matrix = (x @ y.T) / (x.norm(dim=1).view(-1, 1) @ y.norm(dim=1).view(1, -1))

    return cosine_similarity_matrix
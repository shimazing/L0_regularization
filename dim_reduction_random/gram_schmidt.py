import numpy as np


def gs_cofficient(v1, v2):
    return np.dot(v2, v1) / np.dot(v1, v1)

def multiply(cofficient, v):
    return list(map((lambda x : x * cofficient), v))

def proj(v1, v2):
    return multiply(gs_cofficient(v1, v2) , v1)

def gs(X):
    Y = []
    for i in range(len(X)):
        temp_vec = X[i]
        for inY in Y :
            proj_vec = proj(inY, X[i])
            temp_vec = list(map(lambda x, y : x - y, temp_vec, proj_vec))
        temp_vec = np.array(temp_vec)
        temp_vec = temp_vec / np.linalg.norm(temp_vec, 2) # normalize
        Y.append(temp_vec)
    return np.array(Y)

if __name__ == '__main__':
    random_W = np.random.normal(size=(5, 5))
    orthogonal_W = gs(random_W)
    for i in range(5):
        print(i, i, orthogonal_W[i].dot(orthogonal_W[i]))
        for j in range(i+1, 5):
            print(i, j, orthogonal_W[i].dot(orthogonal_W[j]))

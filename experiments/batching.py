import torch

if __name__ == "__main__":

    nx = 2
    nt = 3
    ne = 4

    K = torch.rand((nx, nx))
    X = torch.rand((nx, nt, ne))  # nx nt ne

    # [10 10]
    # [30 20 1 10]
    # KX = (X'K')'
    Y_broadcast = torch.matmul(X[:, None, :, :].T, K.T).T.squeeze(1)

    Y_loop = torch.empty((nx, nt, ne))

    for i in range(nt):
        for j in range(ne):
            Y_loop[:, i, j] = K @ X[:, i, j]

    assert torch.allclose(Y_broadcast, Y_loop)

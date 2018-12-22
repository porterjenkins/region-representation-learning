import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim


class DiscriminatorMLP(nn.Module):
    """
    Discriminator for RegionEncoder model
        - Simple Multilayer Perceptron w/ k layers
        to unify graph/image hidden states into latent
        space
    """
    def __init__(self, x_features, z_features):
        super(DiscriminatorMLP, self).__init__()
        self.W_0 = nn.Linear(x_features + z_features, 16, bias=True)
        self.W_output = nn.Linear(16, 1, bias=True)

    def forward(self, x, z):
        X = torch.cat((x, z), dim=-1)
        h = F.relu(self.W_0(X))
        y_hat = F.sigmoid(self.W_output(h))

        return y_hat

class DiscriminatorNCF(nn.Module):
    """
    Discriminator for RegionEncoder model
        - Model based on Neural Collaborative Filtering
        paper (He et al. WWW17)
        - See eq. 11 for formulation of layer
    """
    def __init__(self, x_features, z_features):
        super(DiscriminatorNCF, self).__init__()
        pass



if __name__ == "__main__":
    n = 100
    p = 32
    mu = np.zeros(p)
    sig = np.eye(p)
    x = np.random.multivariate_normal(mean=mu, cov=sig, size=n)
    z = np.random.multivariate_normal(mean=mu, cov=sig, size=n)

    y = np.zeros(shape=(n, 1), dtype=np.float32)

    for i in range(n):
        if x[i, 0] > 0:
            y[i, 0] = np.random.binomial(1, .9)

    mod = DiscriminatorMLP(x_features=p, z_features=p)

    x_train = torch.from_numpy(x).type(torch.FloatTensor)
    z_train = torch.from_numpy(z).type(torch.FloatTensor)
    y_train = torch.from_numpy(y).type(torch.FloatTensor)

    optimizer = optim.SGD(mod.parameters(), lr=0.1, momentum=0.9)
    cross_entropy = torch.nn.BCELoss()
    n_epoch = 100

    for i in range(n_epoch):
        optimizer.zero_grad()

        # forward + backward + optimize
        y_hat = mod.forward(x_train, z_train)
        loss = cross_entropy(y_hat, y_train)
        loss.backward()
        optimizer.step()

        # print statistics
        loss.item()
        print("Epoch: {}, Train Loss {:.4f}".format(i, loss.item()))
from torch import nn
from libcpab.libcpab.cpab import Cpab


class AlignmentNet(nn.Module):
    def __init__(self, n, k, tesselation_size, circularity, zero_boundary, cpab_device):
        super().__init__()
        self.n = n
        self.k = k
        self.tesselation_size = tesselation_size
        self.circularity = circularity
        self.zero_boundary = zero_boundary
        self.backend = 'pytorch'

        self.transformers = []
        self.theta_dims = []
        for tess_size in self.tesselation_size:
            transformer = Cpab([tess_size, ], backend=self.backend, device=cpab_device,
                               zero_boundary=self.zero_boundary,
                               volume_perservation=False, circularity=self.circularity, override=False)
            self.transformers.append(transformer)
            self.theta_dims.append(transformer.get_theta_dim())

        # Inputs to hidden layer linear transformation
        self.act = nn.Tanh()
        self.pool = nn.MaxPool1d(2)
        self.drop = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 128)

        self.conv1 = nn.Conv1d(self.k, 128, 7, padding=3)
        self.batch_norm1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(128, 64, 5, padding=2)
        self.batch_norm2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 64, 3, padding=1)
        self.batch_norm3 = nn.BatchNorm1d(64)

        self.fc3_thetas = nn.ModuleList()
        self.fc3 = nn.Linear(128, self.theta_dims[0])

        for i in range(len(self.transformers)):
            if i is 0:
                lin = nn.Linear(128, self.theta_dims[i])
            else:
                lin = nn.Linear(200, self.theta_dims[i])
            self.fc3_thetas.append(lin)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x_1 = self.pool(self.act(self.batch_norm1(self.conv1(x))))
        x_1 = self.pool(self.act(self.batch_norm2(self.conv2(x_1))))
        x_1 = self.pool(self.act(self.batch_norm3(self.conv3(x_1))))
        x_1 = x_1.flatten(start_dim=1)
        x_1 = self.act(self.fc1(x_1))
        x_1 = self.act(self.fc2(x_1))
        thetas = []
        output = x
        for i in range(len(self.transformers)):
            theta = nn.Tanh()(self.fc3_thetas[i](x_1))
            thetas.append(theta)
            output = self.transformers[i].transform_data(output, theta, outsize=[self.n])
            x_1 = output.flatten(start_dim=1)
        return output, thetas

    def loss(self, X, y):
        '''
        Torch data format is  [N, C, W] W=timesteps
        Args:
            X: input shapes
            y: input labels
        Returns:
            l2 variance loss
        '''
        loss = 0
        n_classes = y.unique()
        for i in n_classes:
            X_within_class = X[y == i]
            loss_per_class = X_within_class.var(dim=0, unbiased=False).mean(dim=1).mean()
            loss += loss_per_class
        loss /= len(n_classes)
        return loss

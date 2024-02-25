# models for fmgene project
# status: in-developing

import torch.nn as nn
import torch
import sys
from torch.autograd import Variable
import torch.optim as optim
from sklearn.ensemble import VotingClassifier
import numpy as np


class _FCCNN(nn.Module):
    # specifically for dummpy data
    # to be updated when the actual data are available
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv3d(config['in_channels'], config['fil_num_conv'], config['kernel_size'], config['stride'], config['padding'])
        self.conv1_bn = nn.BatchNorm3d(config['fil_num_conv'])
        self.conv2 = nn.Conv3d(config['fil_num_conv'], config['fil_num_conv']//2, config['kernel_size'], config['stride'], config['padding'])
        self.conv2_bn = nn.BatchNorm3d(config['fil_num_conv']//2)
        self.fc1 = nn.Linear(config['fc_insize'], config['fil_num_fc'])
        # self.fc1_bn = nn.BatchNorm1d(config['fil_num_fc'])
        self.fc2 = nn.Linear(config['fil_num_fc'], config['out_dim'])
        self.dr = nn.Dropout(config['dropout'])
        self.a = nn.LeakyReLU()
        self.ao = nn.Sigmoid()

    def forward_(self, fmri, gene):
        # only put in fmri for now
        # print(fmri.shape)
        x_fmri = self.conv1(fmri)
        x_fmri = self.conv1_bn(x_fmri)
        x_fmri = self.dr(self.a(x_fmri))
        # print(x_fmri.shape)
        x_fmri = self.conv2(x_fmri)
        x_fmri = self.conv2_bn(x_fmri)
        x_fmri = self.dr(self.a(x_fmri))
        x_fmri = x_fmri.view(x_fmri.shape[0], -1)
        # print(x_fmri.shape)
        x_fmri = self.fc1(x_fmri)
        # print(x_fmri.shape)
        # x_fmri = self.fc1_bn(x_fmri)
        x_fmri = self.dr(self.a(x_fmri))
        x_fmri = self.fc2(x_fmri)
        x_fmri = self.ao(x_fmri)
        return x_fmri

    # specific for shap only due to it's limitation!
    def forward(self, fmri):
        # only put in fmri for now
        # print(fmri.shape)
        x_fmri = self.conv1(fmri)
        x_fmri = self.conv1_bn(x_fmri)
        x_fmri = self.dr(self.a(x_fmri))
        x_fmri = nn.LeakyReLU()(x_fmri)
        x_fmri = nn.Dropout(p=0.5)(x_fmri)
        # print(x_fmri.shape)
        x_fmri = self.conv2(x_fmri)
        x_fmri = self.conv2_bn(x_fmri)
        x_fmri = nn.LeakyReLU()(x_fmri)
        x_fmri = nn.Dropout(p=0.5)(x_fmri)
        x_fmri = x_fmri.view(x_fmri.shape[0], -1)
        # print(x_fmri.shape)
        x_fmri = self.fc1(x_fmri)
        # print(x_fmri.shape)
        # x_fmri = self.fc1_bn(x_fmri)
        x_fmri = nn.LeakyReLU()(x_fmri)
        x_fmri = nn.Dropout(p=0.5)(x_fmri)
        x_fmri = self.fc2(x_fmri)
        x_fmri = self.ao(x_fmri)
        return x_fmri

class _MLP_gene(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc_layers = nn.Sequential(
                nn.Linear(500, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Sigmoid()
            )
    
    def forward(self, gene):
        output = self.fc_layers(gene)
        return output

    def predict(self, genes):
        results = []
        for gene in genes:
            gene = torch.tensor(gene).to(torch.device('cuda:0' if (torch.cuda.is_available()) else "cpu"), dtype=torch.float)
            gene = gene.view(1, 500)
            hidden = torch.zeros(2, 1, 64)
            hidden = hidden.to(torch.device('cuda:0' if (torch.cuda.is_available()) else "cpu"), dtype=torch.float)
            with torch.no_grad():
                pred = self.forward(gene, hidden)
                pred = pred.squeeze(-1).numpy()
            results.append(pred[0])
        return np.asarray(results)

class _RNN_gene(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc_layers = nn.Sequential(
                nn.Linear(500, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Sigmoid()
            )
    
    def forward(self, gene):
        output = self.fc_layers(gene)
        return output

    def predict(self, genes):
        results = []
        for gene in genes:
            gene = torch.tensor(gene).to(torch.device('cuda:0' if (torch.cuda.is_available()) else "cpu"), dtype=torch.float)
            gene = gene.view(1, 500)
            hidden = torch.zeros(2, 1, 64)
            hidden = hidden.to(torch.device('cuda:0' if (torch.cuda.is_available()) else "cpu"), dtype=torch.float)
            with torch.no_grad():
                pred = self.forward(gene, hidden)
                pred = pred.squeeze(-1).numpy()
            results.append(pred[0])
        return np.asarray(results)

class _RNN_fMRI(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rnn = nn.RNN(input_size=config['input_size'], hidden_size=config['hidden_size'], num_layers=config['num_layers'], batch_first=True)
        self.fc = nn.Linear(config['hidden_size'], config['out_dim'])

    def forward(self, x, gene):
        x = x.flatten(start_dim=2)  # Flatten all dimensions except batch and channels
        x, _ = self.rnn(x)
        x = x[:, -1, :]  # Take the output of the last RNN cell
        x = self.fc(x)
        return torch.sigmoid(x)

class _MLP_fMRI(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config['input_size'], config['fc1_out'])
        self.fc2 = nn.Linear(config['fc1_out'], config['fc2_out'])
        self.fc3 = nn.Linear(config['fc2_out'], config['out_dim'])
        self.relu = nn.ReLU()

    def forward(self, x, gene):
        x = torch.mean(x, 1, keepdim=True) # take average to reduce memory usage.
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)

class _Merged(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv3d(config['in_channels'], config['fil_num_conv'], config['kernel_size'], config['stride'], config['padding'])
        self.conv1_bn = nn.BatchNorm3d(config['fil_num_conv'])
        self.conv2 = nn.Conv3d(config['fil_num_conv'], config['fil_num_conv']//2, config['kernel_size'], config['stride'], config['padding'])
        self.conv2_bn = nn.BatchNorm3d(config['fil_num_conv']//2)
        self.fc1 = nn.Linear(config['fc_insize'], config['fil_num_fc'])
        # self.fc1_bn = nn.BatchNorm1d(config['fil_num_fc'])
        self.fc2 = nn.Linear(config['fil_num_fc'], config['out_dim'])
        self.fc1_gene = nn.Linear(500, 64)
        self.fc2_gene = nn.Linear(64, config['out_dim'])
        self.dr = nn.Dropout(config['dropout'])
        self.fc1_comb = nn.Linear(config['out_dim']*2, 2)
        self.a = nn.LeakyReLU()
        self.ao = nn.Sigmoid()

    def forward(self, fmri, gene):
        # only put in fmri for now
        # print(fmri.shape)
        gene = self.fc1_gene(gene)
        gene = self.dr(self.a(gene))
        gene = self.fc2_gene(gene)
        gene = self.ao(gene)
        # print(gene.shape)
        x_fmri = self.conv1(fmri) 
        x_fmri = self.conv1_bn(x_fmri)
        x_fmri = self.dr(self.a(x_fmri))
        x_fmri = self.conv2(x_fmri)
        x_fmri = self.conv2_bn(x_fmri)
        x_fmri = self.dr(self.a(x_fmri))
        x_fmri = x_fmri.view(x_fmri.shape[0], -1)
        x_fmri = self.fc1(x_fmri)
        # x_fmri = self.fc1_bn(x_fmri)
        x_fmri = self.dr(self.a(x_fmri))
        x_fmri = self.fc2(x_fmri)
        x_fmri = self.ao(x_fmri)
        # print(x_fmri.shape)
        # gene_avg = torch.mean(gene, dim=1)
        # print("Average of gene features:", gene_avg)
        # x_fmri_avg = torch.mean(x_fmri, dim=1)
        # print("Average of fMRI features:", x_fmri_avg)
        # sys.exit()
        combined = torch.cat((x_fmri, gene), dim=1)
        output = self.fc1_comb(combined)
        output = self.ao(output)
        # print(output.shape)
        return output

class BinaryClassifier3D(nn.Module):
    def __init__(self, config):
        super(BinaryClassifier3D, self).__init__()

        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.inorm1 = nn.InstanceNorm3d(32)
        self.pool1 = nn.MaxPool3d(2)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.inorm2 = nn.InstanceNorm3d(64)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.inorm3 = nn.InstanceNorm3d(128)
        self.pool3 = nn.MaxPool3d(2)

        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)
        self.inorm4 = nn.InstanceNorm3d(256)
        self.pool4 = nn.MaxPool3d(2)

        self.conv5 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=2)
        self.inorm5 = nn.InstanceNorm3d(256)
        self.pool5 = nn.MaxPool3d(2)

        self.conv6 = nn.Conv3d(256, 64, kernel_size=1)
        self.inorm6 = nn.InstanceNorm3d(64)
        self.avgpool = nn.AvgPool3d(kernel_size=(3, 3, 1))
        if config['type'] == 'Resting':
            self.avgpool = nn.AvgPool3d(kernel_size=(3, 3, 2))
        self.identity = nn.Identity()
        self.output = nn.Conv3d(64, 2, kernel_size=1)
        self.ao = nn.Sigmoid()

    def forward(self, x, gene):
        # print(x.shape)
        # sys.exit()
        x = torch.mean(x, 1, keepdim=True) # input dim is larger than 1 in some dataset due to time series, take average.
        x = self.conv1(x)
        x = self.inorm1(x)
        x = self.pool1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.inorm2(x)
        x = self.pool2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.inorm3(x)
        x = self.pool3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.inorm4(x)
        x = self.pool4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.inorm5(x)
        x = self.pool5(x)
        x = self.relu(x)

        x = self.conv6(x)
        x = self.inorm6(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = self.identity(x)
        x = self.output(x)
        
        x = x.view(-1,x.shape[1])
        x = self.ao(x)

        return x

if __name__ == '__main__':
    # Example sizes
    size1 = torch.Size([1, 140, 64, 64, 48])
    size2 = torch.Size([1, 105, 64, 64, 24])
    size3 = torch.Size([1, 1, 64, 64, 24])

    # Calculating the flattened size
    input_size1 = 140 * 64 * 64 * 48
    input_size2 = 105 * 64 * 64 * 24
    input_size3 = 1 * 64 * 64 * 24

    # Creating models
    rnn_model1 = RNNModel(input_size1)
    mlp_model1 = MLPModel(input_size1)

    rnn_model2 = RNNModel(input_size2)
    mlp_model2 = MLPModel(input_size2)

    rnn_model3 = RNNModel(input_size3)
    mlp_model3 = MLPModel(input_size3)

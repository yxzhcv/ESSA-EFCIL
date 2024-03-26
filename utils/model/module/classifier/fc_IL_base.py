import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class LinearClassifier(nn.Module):
    def __init__(self, outplanes, args):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(outplanes, args.base_class, bias=True)

    def forward(self, x, num_batch):
        x = self.fc(x)
        return x

    def Incremental_learning(self, numclass):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        self.fc = nn.Linear(in_feature, numclass, bias=True)
        self.fc.weight.data[:out_feature] = weight[:out_feature]
        self.fc.bias.data[:out_feature] = bias[:out_feature]

# SFAN
class SFAN(nn.Module):
    def __init__(self, outplanes, args):
        super(SFAN, self).__init__()
        self.fc1 = nn.Linear(outplanes, 1024)
        self.fc2 = nn.Linear(1024, outplanes)
        self.relu = nn.ReLU(inplace=False)
        
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

        init.constant_(self.fc1.bias, 0)
        init.constant_(self.fc2.bias, 0)
        
    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x += residual
        return x

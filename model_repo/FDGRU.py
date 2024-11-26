import torch
from torch import nn
import os
import torch.nn.functional as F


class Net(nn.Module):
    def __init__ (self, _,inp_size = 1024, hid_size = 1024, seq = 32, layers = 1, nclass = 4, initialization=True):
        super(Net, self).__init__()
        self.name = os.path.basename(__file__).split('.')[0]
        self.inp_size = inp_size
        self.hid_size = hid_size
        self.layers = layers
        self.seq = seq
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3, stride=2,padding=1)

        self.rnn = nn.GRU(
            input_size = inp_size,
            hidden_size = hid_size,
            num_layers = layers,
            bidirectional=False,
            dropout=0.,
            batch_first=True)

        self.linear1 = nn.Linear(self.seq, inp_size)
        self.batch_norm1 = nn.BatchNorm1d(inp_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(inp_size, hid_size)
        self.batch_norm2 = nn.BatchNorm1d(hid_size)
        self.linear3 = nn.Linear(hid_size, nclass)

        self.softmax = nn.Softmax(dim=1)

        if initialization:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LSTM):
                    nn.init.orthogonal_(m.weight_ih_l0)
                    nn.init.orthogonal_(m.weight_hh_l0)
                    nn.init.orthogonal_(m.weight_ih_l1)
                    nn.init.orthogonal_(m.weight_hh_l1)

    def forward(self, x):
        x = x[:,:3,:]
        x= self.conv1(x)
        x = F.pad(x, (137, 137), "constant", 0)
        x.squeeze()
        x = x.view(-1, self.seq)
        x = self.linear1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = x.view(-1, self.seq, self.inp_size)
        h0 = torch.zeros(self.layers, x.size(0), self.hid_size).cuda()
        # c0 = torch.zeros(self.layers, x.size(0), self.hid_size)
        out, _ = self.rnn(x, h0)
        out = out + x
        out = self.linear2(out[:, -1, :])
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear3(out)
        # out = self.softmax(out)
        return out


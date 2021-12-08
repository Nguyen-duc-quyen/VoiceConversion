"""
Define the AutoVC model, including 3 modules:
    - Encoder 
    - Decoder 
    - Postnet
"""


import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias = True, w_init_gain = 'linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias= bias)

        torch.nn.init.xavier_uniform(self.linear_layer.weight, gain= nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding= None, dilation= 1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        
        if padding is None:
            assert(kernel_size%2 == 1)
            padding = int(dilation * (kernel_size - 1)/2)

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation= dilation, bias=bias)

        nn.init.xavier_uniform(self.conv, gain= nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, dim_neck, dim_emb, freq):
        super(Encoder, self).__init__()

        self.dim_neck = dim_neck
        self.freq = freq

        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(80 + dim_emb if i == 0 else 512, 512, kernel_size= 5, stride= 1, padding= 2, dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512)
            )
            convolutions.append(conv_layer)
        
        self.convolutions = nn.ModuleList(convolutions)

        self.ltsm = nn.LTSM(512, dim_neck, 2, batch_first= True, bidirectional= True)

    def forward(self, x, c_org):
        x = x.squeeze(1).transpose(2, 1)
        c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat((x, c_org), dim= 1)

        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2) 

        self.ltsm.flatten_parameters()
        outputs, _ = self.ltsm(x)
        out_forward = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, :self.dim_neck:]

        codes = []

        for i in range(0, outputs.size(1), self.freq):
            codes.append(torch.cat((out_forward[:, i + self.freq - 1, :], out_backward[:, 1, :]), dim=1))

        return codes 

    
class Decoder(nn.Module):
    def __init__(self, dim_neck, dim_emb, dim_pre):
        super(Decoder, self).__init__()

        self.ltsm1 = nn.LTSM(dim_neck*2 + dim_emb, 1, batch_first= True)

        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_pre, dim_pre, kernel_size= 5, stride= 1, padding= 2, dilation= 1, w_init_gain= 'relu'),
                nn.BatchNorm1d(dim_pre)
            )
            convolutions.append(conv_layer)

        self.convolutions = nn.ModuleList(convolutions)

        self.ltsm2 = nn.LTSM(dim_pre, 1024, 2, batch_first= True)

        self.linear_projection = LinearNorm(1024, 88)

    
    def forward(self, x):

        x, _ = self.ltsm1(x)

        x = x.transpose(1, 2)

        for conv in self.convolutions:
            x = F.relu(conv(x))

        x = x.transpose(1,2)

        outputs, _ = self.ltsm2(x)

        decoder_output = self.linear_projection(outputs)

        return decoder_output

class Postnet(nn.Module):
    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(80, 512, kernel_size=5, stride=1, padding=2, dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512)
            )
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
                ConvNorm(512, 512, kernel_size=5, stride=1, padding=2 , dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512)
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(512, 80, kernel_size=5, stride=1, padding=2, dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(80)
            )
        )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)

        return x


class Generator(nn.Module):
    def __init__(self, dim_neck, dim_emb, dim_pre, freq):
        super(Generator, self).__init__()

        self.encoder = Encoder(dim_neck, dim_emb, freq)
        self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        self.postnet = Postnet()

    def forward(self, x, c_org, c_trg):
        codes = self.encoder(x, c_org)

        if c_trg is None:
            return torch.cat(codes, dim=1)

        tmp = []
        for code in codes:
            tmp.append((code.unsqueeze(1).expand(-1, x.size(1)/len(codes), -1)), dim=1)
        
        code_exp = torch.cat(tmp, dim=1)

        encoder_outputs = torch.cat((code_exp, c_trg.unsqueeze(1).expand(1, x.size(1), -1)), dim=1)

        mel_outputs = self.decoder(encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2,1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2, 1)

        mel_outputs = mel_outputs.unsqueeze(1)
        mel_outputs_postnet = mel_outputs_postnet.unsqueeze(1)

        return mel_outputs, mel_outputs_postnet, torch.cat(codes, dim=-1)

import torch
import torch.nn as nn
import torch.nn.functional as F

class HiddenEncoder(nn.Module):
    def __init__(self, in_dim=256, out_dim=128):
        super(HiddenEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, in_dim, 3, padding=1, groups=in_dim)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 1)

    def forward(self, x):
        x = self.conv1(x)
        out = self.conv2(x)
        return out
 
class TempSoftmax(torch.nn.Module):
    def __init__(self, initial_temp=1.0):
        super(TempSoftmax, self).__init__()
        self.temperature = torch.nn.Parameter(torch.tensor(initial_temp), requires_grad=True)
    
    def forward(self, input):
        scaled_input = input / self.temperature
        softmax_output = F.softmax(scaled_input, dim=2)
        return softmax_output


class MotionEncoder(nn.Module):
    def __init__(self, args):
        super(MotionEncoder, self).__init__()
        flo_dim = 16
        cor_dim = args.corr_levels * (2*args.corr_radius + 1)**2 # 147, 243

        self.flo_conv = nn.Conv2d(2, flo_dim, 3, padding=1)
        self.mot_conv = nn.Sequential(
            HiddenEncoder(in_dim=cor_dim+flo_dim, out_dim=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 94, 3, padding=1))
        # self.mot_conv = HiddenEncoder(in_dim=cor_dim+flo_dim, out_dim=hidden_dim-2)

    def forward(self, flow, corr):
        flo = F.relu(self.flo_conv(flow))              #  2  --> 13
        cor_flo = torch.cat([corr, flo], dim=1) # 243  + 13
        out = self.mot_conv(cor_flo)          # 256 --> 62

        return torch.cat([out, flow], dim=1)   # 62   +  2


class FlowHead(nn.Module):
    def __init__(self, input_dim=64, mid_dim=32):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, mid_dim, 3, padding=1)
        # self.conv1 = HiddenEncoder(input_dim, out_dim=mid_dim)
        self.conv2 = nn.Conv2d(mid_dim, 2, 3, padding=1)
        # self.conv3 = nn.Conv2d(16, 2, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        x = self.conv2(x)
        return x


class CGC(nn.Module):
    def __init__(self, hidden_dim=64, input_dim=128):
        super(CGC, self).__init__()
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convz = nn.Conv2d(hidden_dim+hidden_dim, hidden_dim, 3, padding=1)
        self.convh = nn.Conv2d(hidden_dim+hidden_dim, hidden_dim, 3, padding=1)
    def forward(self, h, x):

        hx = torch.cat([h, x], dim=1)
        ch = torch.sigmoid((self.convr(hx)))
        q = torch.cat([ch, h], dim=1)  
        z = torch.sigmoid(self.convz(q))     
        h = self.convh(torch.cat([z, ch], dim=1))
        h = torch.tanh(h)
        return h

class SmallUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=64):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = MotionEncoder(args)                              
        self.gru = CGC(hidden_dim=hidden_dim, input_dim=128)
        self.flow_head = FlowHead(input_dim=hidden_dim, mid_dim=64)
        self.mask = HiddenEncoder(in_dim=hidden_dim, out_dim=64*9)
        # self.mask = nn.Conv2d(hidden_dim, 16*9, 3, padding=1)


    def forward(self, net, inp, corr, flow, itr, iters):
        motion_features = self.encoder(flow, corr)          # 2 + 273 --> 96
        inp = torch.cat([inp, motion_features], dim=1)      # 64 + 96 --> 160 
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        if itr == iters - 1:
            mask = self.mask(net)
        else:
            mask =None

        return net, mask, delta_flow





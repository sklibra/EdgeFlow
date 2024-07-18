from pickle import FALSE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.update import SmallUpdateBlock, HiddenEncoder, TempSoftmax
from core.lmacnet import LMAC_tiny
from core.corr import CorrBlock
from core.utils.utils import coords_grid, upflow16, upflow2

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class EdgeFlow(nn.Module):
    def __init__(self, args):
        super(EdgeFlow, self).__init__()
        self.args = args
        self.hidden_dim = hdim = 64
        self.context_dim = cdim = 32
        args.corr_levels = 3
        args.corr_radius = 3

        self.fnet = LMAC_tiny(out_c=128) 
        self.cnet = HiddenEncoder(in_dim=128, out_dim=hdim+cdim)
        self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)
        self.tempsoftmax = TempSoftmax(initial_temp=1.0)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//16, W//16, device=img.device)
        coords1 = coords_grid(N, H//16, W//16, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        # mask = torch.softmax(mask, dim=2)
        mask = self.tempsoftmax(mask)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, flow_init=None, iters=3):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()
        # print(image1.shape)

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])        
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()


        corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.args.corr_levels, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(fmap1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        #flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow, itr, iters)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if itr == iters - 1:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
                flow_up = upflow2(flow_up)

        return coords1 - coords0, flow_up

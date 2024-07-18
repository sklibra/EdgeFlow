# -*- coding: utf-8 -*-
import time
import torch
import sys
sys.path.append('core')
import argparse
import torch

from core.edgeflow import EdgeFlow
from core.utils import flow_viz
from core.utils.utils import InputPadder
import numpy as np
import cv2
from PIL import Image
from core.utils.utils import InputPadder
from core.utils import flow_viz

# def load_image(imfile):
#     img = np.array(Image.open(imfile)).astype(np.uint8)
#     img = torch.from_numpy(img).permute(2, 0, 1).float()
#     return img[None].to('cuda')


def demo(args):
    model = torch.nn.DataParallel(EdgeFlow(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.cuda()
    model.eval()


    # image1 = load_image('frame_0016.png')
    # image2 = load_image('frame_0017.png')
    # padder = InputPadder(image1.shape)
    # image1, image2 = padder.pad(image1, image2)
    # image1 = torch.randn(1, 3, 448, 1024).cuda()
    # image2 = torch.randn(1, 3, 448, 1024).cuda()

    # image1 = torch.randn(1, 3, 480, 640).cuda()
    # image2 = torch.randn(1, 3, 480, 640).cuda()
    # torch.onnx.export(model,
    #                   (image1, image2),
    #                   'EF640_4.onnx',
    #                   export_params=True,
    #                   verbose=False,
    #                   input_names=['data1', 'data2'],
    #                   output_names=['flow_low', 'flow_up'],
    #                   opset_version=11
    #                   )
# Test hot start
    image1 = torch.randn(1, 3, 480, 640).cuda()
    image2 = torch.randn(1, 3, 480, 640).cuda()
    flow_init = torch.randn(1, 2, 30, 40).cuda()  
    output_names = ['flow_low', 'flow_up']
    torch.onnx.export(model,
                      (image1, image2, flow_init),
                      'A640_3.onnx',
                      export_params=True,
                      verbose=False,
                      input_names=['data1', 'data2','data3'],
                      output_names=['flow_low', 'flow_up'],
                      opset_version=11
                      )

# onnxsim EF1024_4.onnx sim1024_4.onnx                      


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="models/edgev2-sintel.pth")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    args = parser.parse_args()

    demo(args)
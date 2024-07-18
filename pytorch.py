import time
import torch
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
# from calflops import calculate_flops

from core.edgeflow import EdgeFlow
from core.utils import flow_viz
from core.utils.utils import InputPadder, forward_interpolate

def show_flow_hsv(flow, show_style=2):

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])#将直角坐标系光流场转成极坐标系
    hsv      = np.zeros((flow.shape[0], flow.shape[1], 3), np.uint8)

    #Color patterns for optical flow visualization
    if show_style == 1:
        hsv[..., 0] = ang * 180 / np.pi / 2 #angle弧度转角度
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)#magnitude归到0～255之间
    elif show_style == 2:
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = 255

    #From hsv to bgr
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


DEVICE = 'cuda'

# Image loading
def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to('cuda')
# Optical flow visualization
def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    # img = img[:, :, [2,1,0]]
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    # flo = show_flow_hsv(flo)
    img_flo = np.concatenate([img, flo], axis=0)
    
    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.imshow('image', img_flo/255.0)
    cv2.waitKey()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def demo(args):
    model = torch.nn.DataParallel(EdgeFlow(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        input_t = torch.randn(1, 3, 480, 640).cuda()
        print(input_t.shape)
        padder = InputPadder(input_t.shape)
        image1, image2 = padder.pad(input_t, input_t)
        flow_low, flow_up = model(image1, image2, iters=4)
        print(flow_up.shape)
        
        # # Image optical flow visualization
        # image1 = load_image('frame_0016.png')
        # image2 = load_image('frame_0017.png')
        # padder = InputPadder(image1.shape)
        # image1, image2 = padder.pad(image1, image2)
        # flow_low, flow_up = model(image1, image2, iters=4)
        # viz(image1, flow_up)

        # # preheat
        for x in range(100):
            flow_low, flow_up = model(image1, image2, iters=4)

        # # Speed test
        iter = 1000
        start = time.time()
        # flow_prev = None
        for x in range(iter):
            flow_low, flow_up = model(image1, image2, iters=4)
        #    warm_upOptical flow initialization
            # flow_low, flow_up = model(image1, image2, iters=4, flow_init=flow_prev)
            # flow_prev = forward_interpolate(flow_low[0])[None].cuda()
        end = time.time()

        # print(flow_prev.shape)
        mean_time = 1000. * (end-start)/iter
        print('Time elapsed: {:.3f} ms'.format(mean_time))
        mean_fps = 1000. / mean_time
        print(' FPS: {mean_fps:.2f}'.format(mean_fps=mean_fps))
        model = model.train()
        print('Number of parameters: {:.2f} M'.format(count_parameters(model) / 1e6))


        # MACs computation
        # image1 = torch.randn(1, 3, 448, 1024).cuda()
        # image2 = torch.randn(1, 3, 448, 1024).cuda()
        # iter   = torch.tensor(4).cuda()
        # inputs = {"image1": image1, "image2": image2, "iters": iter}
        # flops, macs, params = calculate_flops(model=model, kwargs=inputs)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="models/edgev2-things.pth")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    args = parser.parse_args()

    demo(args)
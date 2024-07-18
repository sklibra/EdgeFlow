import torch
import time
import onnx
import onnxruntime
import numpy as np
import cv2
from PIL import Image
from core.utils.utils import InputPadder
from core.utils import flow_viz
# from core.utils.utils import forward_interpolate


# Image loading
def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to('cuda')
# Optical flow visualization
def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)
    
    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()

# Load the ONNX model and perform the inference
def inference_onnx_model():

    onnx_model = onnx.load("EF1024_4.onnx")
    ort_session = onnxruntime.InferenceSession(onnx_model.SerializeToString(), providers=['CUDAExecutionProvider'])

    image1 = load_image('frame_0016.png')
    image2 = load_image('frame_0017.png')
    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)
    image1 = image1.cpu().numpy()
    image2 = image2.cpu().numpy()

    # Perform model reasoning and output visual pictures
    # flow_low, flow_up = ort_session.run(['flow_low', 'flow_up'], {'data1': image1, 'data2': image2})
    # print(flow_up.shape)
    # viz(torch.from_numpy(image1), torch.from_numpy(flow_up))

    # Inference speed test
    # Early speed is not stable, preheat first
    for x in range(300):
        outputs = ort_session.run(['flow_low', 'flow_up'], {'data1': image1, 'data2': image2})

    start = time.time()
    iter = 1000
    for x in range(iter):
        outputs = ort_session.run(['flow_low', 'flow_up'], {'data1': image1, 'data2': image2})
    end = time.time()
    mean_time = 1000. * (end-start)/iter
    print('Time elapsed: {:.3f} ms'.format(mean_time))
    mean_fps = 1000. / mean_time
    print(' FPS: {mean_fps:.2f}'.format(mean_fps=mean_fps))

if __name__ == '__main__':

    inference_onnx_model()

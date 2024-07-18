import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import torch
from PIL import Image
from core.utils.utils import InputPadder
import pycuda.autoinit
from core.utils import flow_viz


def engine_infer(engine, input_image_1, input_image_2):

    context = engine.create_execution_context()
    inputs  = [input_image_1, input_image_2]
    inputs  = [input_tensor.cpu().numpy() if input_tensor.is_cuda else input_tensor.numpy() for input_tensor in inputs]
    inputs  = [np.ascontiguousarray(input_tensor) for input_tensor in inputs]

    d_inputs  = [cuda.mem_alloc(inp.nbytes) for inp in inputs]
    outputs_shape_dtype = [(engine.get_binding_shape(i), engine.get_binding_dtype(i)) for i in range(engine.num_bindings) if not engine.binding_is_input(i)]
    d_outputs = [cuda.mem_alloc(trt.volume(shape) * np.dtype(trt.nptype(dtype)).itemsize) for shape, dtype in outputs_shape_dtype]

    stream    = cuda.Stream()

    for d_input, inp in zip(d_inputs, inputs):
        cuda.memcpy_htod_async(d_input, inp, stream)

    bindings = [int(d_inp) for d_inp in d_inputs] + [int(d_out) for d_out in d_outputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    outputs  = [np.empty(shape, dtype=trt.nptype(dtype)) for shape, dtype in outputs_shape_dtype]
    for d_output, out in zip(d_outputs, outputs):
        cuda.memcpy_dtoh_async(out, d_output, stream)
    stream.synchronize()

    return outputs

DEVICE = 'cuda'
def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)
    
    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()
    cv2.imwrite('image/flow_44.png', flo[:, :, [2,1,0]])
    

image1 = load_image('image/frame_0016.png')
image2 = load_image('image/frame_0017.png')

padder = InputPadder(image1.shape)
image1, image2 = padder.pad(image1, image2)

# Loading engine
TRT_LOGGER = trt.Logger()
engine_file_path = "trt/1024_4.trt"
with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

output = engine_infer(engine, image1, image2)
# visualization
viz(image1, torch.from_numpy(output[1]))
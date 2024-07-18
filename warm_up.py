import cv2
import time
import torch
import numpy as np
import tensorrt as trt
import pycuda.autoinit
from core.utils import flow_viz
from openni import openni2
import pycuda.driver as cuda
from scipy import interpolate

def forward_interpolate(flow):

    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return flow

def show_flow_hsv(flow, show_style=2):

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])#The optical flow field in rectangular coordinate system is transformed into polar coordinate system
    hsv      = np.zeros((flow.shape[0], flow.shape[1], 3), np.uint8)

    #Color patterns for optical flow visualization
    if show_style == 1:
        hsv[..., 0] = ang * 180 / np.pi / 2 #angle Radian Angle of rotation
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)#The magnitude is between 0 and 255
    elif show_style == 2:
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = 255

    #From hsv to bgr
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


DEVICE = 'cuda'
# Load the TensorRT engine
TRT_LOGGER = trt.Logger()
engine_file_path = "trt/640_4.trt"
with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

inputs_shape_dtype  = [(engine.get_binding_shape(i), engine.get_binding_dtype(i)) for i in range(engine.num_bindings) if engine.binding_is_input(i)]
d_inputs  = [cuda.mem_alloc(trt.volume(shape) * np.dtype(trt.nptype(dtype)).itemsize) for shape, dtype in inputs_shape_dtype]
outputs_shape_dtype = [(engine.get_binding_shape(i), engine.get_binding_dtype(i)) for i in range(engine.num_bindings) if not engine.binding_is_input(i)]
d_outputs = [cuda.mem_alloc(trt.volume(shape) * np.dtype(trt.nptype(dtype)).itemsize) for shape, dtype in outputs_shape_dtype]
h_outputs = [np.empty(shape, dtype=trt.nptype(dtype)) for shape, dtype in outputs_shape_dtype]

# Initialize the camera
openni2.initialize()
dev = openni2.Device.open_any()
print(dev.get_device_info())
cap = cv2.VideoCapture(0)

print(f"摄像头的FPS是: {cap.get(cv2.CAP_PROP_FPS)}")# Read the camera's FPS

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# frame_count = 0  # Used to control display frequency and evaluate performance
# time_accum = 0  # Cumulative processing time, used to calculate FPS

stream  = cuda.Stream()

ret, frame = cap.read()
if not ret:
    print("无法获取图像帧！")
image1 = np.transpose(frame, (2, 0, 1)).astype(np.float32)[np.newaxis, :]
flow_init = np.random.randn(1, 2, 30, 40).astype(np.float32)


while True:

    
    ret, frame = cap.read()

    image2 = np.transpose(frame, (2, 0, 1)).astype(np.float32)[np.newaxis, :]
    inputs = [np.ascontiguousarray(image) for image in [image1, image2, flow_init]]

    for d_input, inp in zip(d_inputs, inputs):
        cuda.memcpy_htod_async(d_input, inp, stream)

    # start = time.time()
    context  = engine.create_execution_context()
    bindings = [int(d_inp) for d_inp in d_inputs] + [int(d_out) for d_out in d_outputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # end = time.time()
    # time_accum += (end - start)

    # # Calculate the average processing time and estimate FPS every 100 frames
    # if frame_count % 100 == 0 and frame_count != 0:
    #     avg_time_per_frame = time_accum / 100
    #     fps = 1 / avg_time_per_frame
    #     print('Average FPS: {:.2f}'.format(fps))
    #     time_accum = 0  # Reset accumulation time

    for d_output, out in zip(d_outputs, h_outputs):
        cuda.memcpy_dtoh_async(out, d_output, stream)
    stream.synchronize()

    flow_init = np.squeeze(h_outputs[1])
    # print(np.shape(flow_init))
    flow_init = forward_interpolate(flow_init)
    flow_init = flow_init[np.newaxis, :]
    flow = np.squeeze(h_outputs[0]).transpose(1, 2, 0)
    flow_up = show_flow_hsv(flow)

    img_flo = np.concatenate([frame, flow_up], axis=1)
    # img_flo = cv2.resize(img_flo, (960, 360))
    cv2.imshow('flow', img_flo)
    
    key = cv2.waitKey(1)
    if int(key) == ord('q'):
        break
    image1 = image2

    # end = time.time()
    # print('Time elapsed: {:.3f} s'.format(end - start))

    # frame_count += 1

# Release resources
cap.release()
cv2.destroyAllWindows()
dev.close()

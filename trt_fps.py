import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import torch

import time
import pycuda.autoinit


def engine_infer(engine, input_image_1, input_image_2):

    context = engine.create_execution_context()
    inputs = [input_image_1, input_image_2]
    inputs = [input_tensor.cpu().numpy() if input_tensor.is_cuda else input_tensor.numpy() for input_tensor in inputs]
    inputs = [np.ascontiguousarray(input_tensor) for input_tensor in inputs]

    d_inputs = [cuda.mem_alloc(inp.nbytes) for inp in inputs]
    outputs_shape_dtype = [(engine.get_binding_shape(i), engine.get_binding_dtype(i)) for i in range(engine.num_bindings) if not engine.binding_is_input(i)]
    d_outputs = [cuda.mem_alloc(trt.volume(shape) * np.dtype(trt.nptype(dtype)).itemsize) for shape, dtype in outputs_shape_dtype]

    stream = cuda.Stream()

    for d_input, inp in zip(d_inputs, inputs):
        cuda.memcpy_htod_async(d_input, inp, stream)

    # Preheating process
    for _ in range(100):
        bindings = [int(d_inp) for d_inp in d_inputs] + [int(d_out) for d_out in d_outputs]
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # The initialization variable is used to record the time
    times = []

    # Test inference time
    iter = 100
    for _ in range(iter):
        start = time.time()  # Inferred start time
        bindings = [int(d_inp) for d_inp in d_inputs] + [int(d_out) for d_out in d_outputs]
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        stream.synchronize()  # Ensure that the current inference is complete
        end = time.time()  # Inferred end time
        times.append(1000. * (end - start))  # Convert the time to milliseconds and record it

    mean_time = np.mean(times)  # Calculating mean time
    print('Time elapsed: {:.3f} ms'.format(mean_time))
    mean_fps = 1000. / mean_time
    print('FPS: {:.2f}'.format(mean_fps))


# image1 = torch.randn(1, 3, 448, 1024).cuda()
# image2 = torch.randn(1, 3, 448, 1024).cuda()
image1 = torch.randn(1, 3, 480, 640).cuda()
image2 = torch.randn(1, 3, 480, 640).cuda()
# Loading engine
TRT_LOGGER = trt.Logger()
engine_file_path = "trt/640_3.trt"
with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

engine_infer(engine, image1, image2)
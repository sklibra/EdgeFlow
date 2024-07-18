import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
 
# Load the ONNX file
onnx_file_path = 'onnx/sim1024_4.onnx'
engine_file_path = '1024_4.trt'
 
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
# flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
network = builder.create_network(1)
parser = trt.OnnxParser(network, TRT_LOGGER)
 
# Parse the ONNX file
with open(onnx_file_path, 'rb') as f:
    data = f.read()
    parser.parse(data)
 
# Build the TensorRT engine
config = builder.create_builder_config()
config.max_workspace_size = 4*(1 << 30)

# Build the TensorRT engine
# builder_config = builder.create_builder_config()
# builder_config.max_workspace_size = 1 << 30
# config.max_batch_size = 1  # 设置最大批量大小
# builder_config.set_flag(trt.BuilderFlag.FP16) 
# # builder_config.set_flag(trt.BuilderFlag.INT8) 
# engine = builder.build_engine(network, builder_config)

# Check whether FP16 inference is supported
if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)

engine = builder.build_engine(network, config)
 
# 保存TensorRT引擎到文件
with open(engine_file_path, 'wb') as f:

    f.write(engine.serialize())
 
# ry to use the ONNX-TensorRT library as much as possible
# After installation, run the command:
#  onnx2trt ./onnx/sim1024_4.onnx -o ./trt/sim1024_4.trt -d 16
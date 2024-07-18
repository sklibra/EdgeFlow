<<<<<<< HEAD
# EdgeFlow
=======
<<<<<<< HEAD
# EdgeFlow
=======
# Fast Recurrent Field Transforms for Optical Flow on Edge GPUs



To test pytorch speed and visualization, you can run
<pre><code>$ pytorch.py</code></pre>

To convert pytorch to onnx, you can run
<pre><code>$torch2onnx.py</code></pre>

To test the onnx model for speed and visualization, you can run
<pre><code>$ onnx_time.py</code></pre>

To convert the ONNX model to the TensorRT model, you can run
<pre><code>$ onnx2trt.py</code></pre>
There are two ways to do this:
First,Try to use the ONNX-TensorRT library as much as possible
Second,After installation, run the command:onnx2trt ./onnx/sim1024_4.onnx -o ./trt/sim1024_4.trt -d 16

To benchmark TensorRT inference speed, you can run
<pre><code>$trt_fps.py</code></pre>

To test the TensorRT visual code, you can run
<pre><code>$trt.py</code></pre>

To run the optical flow synchronized visualization using the astra camera, you can run
<pre><code>$warm_up.py</code></pre>



## Support TensorRT

Support TensorRT with below configuration:

Test equipment: Jetson Xavier NX
NVIDIA Software Suite: Jetpack 4.6.4 [L4T 32.7.4]\
cuda: 11.3\
Deep Learning Library: cuDNN 8.2\
Deep Learning Inference Engine: TensorRT 8.2
Deep learning framework: PyTorch 1.6.0
Programming language: Python 3.6.9

>>>>>>> 03ec2ab (first commit)
>>>>>>> ea50d77 (first commit)

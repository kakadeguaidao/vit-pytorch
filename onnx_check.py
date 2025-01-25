import torch
import onnx
import onnxruntime
import time
import numpy as np

onnx_model = onnx.load("vit_model.onnx")
onnx.checker.check_model("vit_model.onnx")

ort_session = onnxruntime.InferenceSession("vit_model.onnx", providers=["CUDAExecutionProvider"])
for item in ort_session.get_inputs():
    print(item)
for item in ort_session.get_outputs():
    print(item)


device = torch.device("cuda:0")

# speed testing
input_images = torch.randn(1, 3, 256, 256)
input_images_np = input_images.cpu().numpy()
total_time = 0.0
output = ort_session.run(None, {"input": input_images_np})
for i in range(10000):
    start_time = time.time()
    output = ort_session.run(None, {"input": input_images_np})
    total_time += time.time() - start_time

print("Average time: ", round(total_time,4))
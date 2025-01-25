import torch
import numpy as np
import tensorrt as trt
import argparse
import time
from common_runtime import allocate_buffers
from common import do_inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tensorrt inference")
    parser.add_argument("--engine_path", type=str, help="trt engine file path")
    
    args = parser.parse_args()

    engine_path = args.engine_path
    TRT_LOGGER = trt.Logger()

    with open(engine_path, "rb") as f:
        engine_byte_string = f.read()
    image = torch.randn(1, 3, 256, 256).numpy()
    # engine obj initiation
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_byte_string)
    context = engine.create_execution_context()
    
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    
    inputs[0].host = image
    
    _ = do_inference(
            context,
            engine=engine,
            bindings=bindings,
            inputs=inputs,
            outputs=outputs,
            stream=stream,
        )
    
    # speed test
    start_time = time.time()
    for i in range(10000):
        # inputs[0].host = image
        _ = do_inference(
            context,
            engine=engine,
            bindings=bindings,
            inputs=inputs,
            outputs=outputs,
            stream=stream,
        )
    
    print(f"10000 inference time: {round(time.time() - start_time, 3)}")
    
    # print(trt_outputs)
    
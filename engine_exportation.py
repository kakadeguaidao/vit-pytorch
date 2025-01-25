import tensorrt as trt
import torch
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export a PyTorch model to TensorRT')
    parser.add_argument('--onnx_path', type=str, help='Path to the PyTorch model')
    parser.add_argument('--output_dir', type=str, help='Path to the output TensorRT engine')
    args = parser.parse_args()

    onnx_path = args.onnx_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # trt config
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    # Load the PyTorch model
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) #we have enabled the explicit Batch
    network = builder.create_network(EXPLICIT_BATCH)
    parser  = trt.OnnxParser(network, logger)

    config.default_device_type = trt.DeviceType.GPU

    # FP16
    # config.flags |= 1 << int(trt.BuilderFlag.FP16) # 有问题，速度没有更快，精度损失很大

    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())


    # 查看 network 中的输入输出
    print("network num inputs: ", network.num_inputs)
    print("network num outputs: ", network.num_outputs)
    for i in range(network.num_inputs):
        print("input: ", network.get_input(i).name, ": ", network.get_input(i).shape)
    for i in range(network.num_outputs):
        print("output: ", network.get_output(i).name, ": ", network.get_output(i).shape)
        
    # Set profile
    profile = builder.create_optimization_profile()
    profile.set_shape("input", 
                    min = (1, 3, 256, 256),
                    opt = (1, 3, 256, 256),
                    max = (1, 3, 256, 256))
    
    
    config.add_optimization_profile(profile)

    engine_string = builder.build_serialized_network(network, config)
    engine_path = os.path.join(output_dir, os.path.basename(onnx_path).replace('.onnx', '.engine'))
    with open(engine_path, 'wb') as f:
        f.write(bytearray(engine_string))
        
    # Create a TensorRT engine
    # with trt.Builder() as builder, builder.create_network() as network, trt.OnnxParser(network, trt.Logger()) as parser:
    #     builder.max_workspace_size = 1 << 30
    #     builder.max_batch_size = 1
    #     parser.parse(model)

    #     with open(args.output, 'wb') as f:
    #         f.write(builder.build_cuda_engine(network))

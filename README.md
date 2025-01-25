## TensorRT、ONNX、Torch速度测试

### 总结

模型：

[vit-pytorch](https://github.com/lucidrains/vit-pytorch	"vit-pytorch")

模型参数：

```json
{
"image_size": 256,
"patch_size": 32,
"num_classes": 10,
"dim": 512,
"depth": 4,
"heads": 4,
"mlp_dim": 512,
"dropout": 0.1,
"emb_dropout": 0.1
}
```

环境：WSL2-Ubuntu2204

硬件：4060Ti

| 输入：tensor[1, 3, 256, 256] | 推理10000次耗时 |  精度   |
| :--------------------------: | :-------------: | :-----: |
|            torch             |     25.85s      | float32 |
|             onnx             |     13.41s      | float32 |
|           tensorrt           |      7.21s      | float32 |

### 相关脚本

（相关代码都需要切换到）

模型训练和推理：train.ipynb

onnx模型导出：onnx_exportation.py

onnx模型推理：onnx_check.py

trt engine导出：engine_exportation.py

trt engine推理：engine_inference_test.py

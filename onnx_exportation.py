import torch
from vit_pytorch import ViT
model = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 10,
    dim = 512,
    depth = 4,
    heads = 4,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
model.load_state_dict(torch.load('checkpoints/vit_4_30000.pth', map_location='cpu'))
model.eval()

output_logits = model(torch.randn(1, 3, 256, 256))
print(output_logits.shape)


input_images = torch.randn(1, 3, 256, 256)

torch.onnx.export(model, (input_images),
                  "vit_model.onnx",
                  verbose=True, export_params=True, do_constant_folding=True,
                  input_names = ["input"],
                  output_names=["output",],
                  dynamic_axes={
                        "input": {0: 'batch_size'},
                        "output": {0: 'batch_size'}
                      })
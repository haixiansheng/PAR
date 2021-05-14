import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx

d = torch.rand(1, 3, 256, 192)
m = model()
o = m(d)
 
onnx_path = "onnx_model_name.onnx"
torch.onnx.export(m, d, onnx_path)
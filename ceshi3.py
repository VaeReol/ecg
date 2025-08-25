import torch
ckpt_path = "/ssd1/qinzehao/new/v5/pretrain/checkpoint-50.pth"
model = torch.load(ckpt_path, map_location='cpu')
print(model['optimizer'])
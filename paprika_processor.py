import os
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from AnimeGANV2 import AnimeGANV2

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def load_image(image_path, x32=False):
    img = Image.open(image_path).convert("RGB")

    if x32:
        def to_32s(x):
            return 256 if x < 256 else x - x % 32
        w, h = img.size
        img = img.resize((to_32s(w), to_32s(h)))

    return img

def process_image(input_image, model, device, upsample_align=True):
    try:
        # 预处理
        image = load_image(input_image, x32=True)  # 转换为 32 的倍数
        image_tensor = to_tensor(image).unsqueeze(0) * 2 - 1  # 归一化到 [-1, 1]
        
        with torch.no_grad():
            out = model(image_tensor.to(device), upsample_align).cpu()
            out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5  # 反归一化到 [0, 1]
            out = to_pil_image(out)
        
        return out
    except Exception as e:
        print(f"处理图像失败: {str(e)}")
        return None
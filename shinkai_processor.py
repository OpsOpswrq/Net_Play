import cv2
import numpy as np
import torch

def process_image(input_image, model, device):
    try:
        # 调整大小并预处理
        img = cv2.resize(input_image, (512, 512))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)) / 255.0  # HWC to CHW
        img = img * 2 - 1  # 归一化到 [-1, 1]
        
        # 使用 FP32（假设 Shinkai.pt 使用 FP32）
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            cartoon = model(img_tensor)
        
        cartoon = cartoon.squeeze(0).cpu().float().numpy().transpose((1, 2, 0))
        cartoon = (cartoon + 1) / 2 * 255
        cartoon = cartoon.astype(np.uint8)
        cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
        return cartoon
    except Exception as e:
        print(f"处理图像失败: {str(e)}")
        return None
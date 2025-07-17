import torch
from PIL import Image
from torchvision import transforms

# 定义均值和标准差
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(means, stds)
])

def process_image(input_image, model, device):
    try:
        # 确保模型的所有参数和缓冲区在正确设备上
        model = model.to(device)
        for param in model.parameters():
            param.data = param.data.to(device)
        for buffer in model.buffers():
            buffer.data = buffer.data.to(device)
        
        # 转换为图像
        t_stds = torch.tensor(stds, device=device, dtype=torch.float16)[:, None, None]
        t_means = torch.tensor(means, device=device, dtype=torch.float16)[:, None, None]
        
        # 转换和预处理图像
        transformed_image = img_transforms(input_image)[None, ...].to(device, dtype=torch.float16)
        
        with torch.no_grad():
            result_image = model(transformed_image)[0]
            output_image = result_image.mul(t_stds).add(t_means).mul(255.).clamp(0, 255).permute(1, 2, 0)
            output_image = output_image.detach().cpu().numpy().astype('uint8')
            output_image = Image.fromarray(output_image)
        return output_image
    except Exception as e:
        print(f"处理图像失败: {str(e)}")
        return None
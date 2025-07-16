import gradio as gr
import mobilenetv3ncnn
import numpy as np
import cv2
from fastrtc import WebRTC

# 这个文件是建立了一个简易的界面来进行操作，有趣味性
# 全局配置
prob_threshold = 0.6

# 在模块顶层加载 NCNN 模型
bin_path = "./params/mobilenetv3-TSC.bin"
param_path = "./params/mobilenetv3-TSC.param"
net = mobilenetv3ncnn.load_ncnn_model(param_path=param_path, bin_path=bin_path, use_gpu=False)

def lauch_app(image):
    # 确保图像是 NumPy 数组并转换为 BGR 格式
    if not isinstance(image, np.ndarray):
        image = np.array(image, dtype=np.uint8)
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = np.ascontiguousarray(image)  # 确保数组连续
    objects = mobilenetv3ncnn.detect(net, image, prob_threshold)
    if objects:
        img = mobilenetv3ncnn.draw(image, objects)
    else:
        img = image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换回 RGB
    return img

if __name__ == "__main__":
    with gr.Blocks() as mobilenetv3_app:
        gr.HTML(
        """
            <h1 style='text-align: center'>
                交通标志识别 - 基于MobileNetV3模型
            </h1>
        """
        )
        with gr.Column():
            with gr.Group():
                video = WebRTC(
                    modality="video",
                    mode="send-receive"
                )
            video.stream(
                fn=lauch_app,
                inputs=[video],
                outputs=[video],
            )
    # share=True 它用来分享
    mobilenetv3_app.launch()
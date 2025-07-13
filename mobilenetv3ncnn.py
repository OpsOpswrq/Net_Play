import ncnn
import numpy as np
import cv2 as cv
import argparse
import torch

# 全局配置，与 mobilenetv3.h 一致
image_size = [320, 320]
num_classes = 58
mean_vals = [123.675, 116.28, 103.53]
norm_vals = [1.0/58.395, 1.0/57.12, 1.0/57.375]

class_names = [
    "speed_five", "speed_fifteen", "speed_thirty", "speed_forty", "speed_fifty",
    "speed_sixty", "speed_seventy", "speed_eighty", "forbid_straight_left", "forbid_straight_right",
    "forbid_straight", "forbid_left", "forbid_left_right", "forbid_right", "forbid_overtake",
    "forbid_turn_around", "forbid_vehicle", "forbid_ring", "release_speed_limit_forty", "release_speed_limit_fifty",
    "straight_right", "straight", "left", "left_right", "right", "lean_left",
    "lean_right", "roundabout", "vehicle", "ring", "non_motor_vehicle",
    "turn_around", "detour_left_and_right", "notice_traffic_sign", "notice_danger", "notice_pedestrian",
    "notice_non_motor_vehicle", "notice_children", "sharp_right_turn", "sharp_left_turn", "downhill",
    "uphill", "slow_down", "right_T_crossroad", "left_T_crossroad", "village",
    "reverse_detour", "unattended_railway_crossings", "under_construction", "continuous_detour", "attended_railway_crossings",
    "accident_prone", "stop_sign", "forbid_through_traffic_sign", "forbid_parking", "forbid_entry",
    "slow_down_and_give_way", "stop_for_inspection"
]

color_list = [
    [216, 82, 24], [236, 176, 31], [125, 46, 141], [118, 171, 47], [76, 189, 237],
    [238, 19, 46], [76, 76, 76], [153, 153, 153], [255, 0, 0], [255, 127, 0],
    [190, 190, 0], [0, 255, 0], [0, 0, 255], [170, 0, 255], [84, 84, 0],
    [84, 170, 0], [84, 255, 0], [170, 84, 0], [170, 170, 0], [170, 255, 0],
    [255, 84, 0], [255, 170, 0], [255, 255, 0], [0, 84, 127], [0, 170, 127],
    [0, 255, 127], [84, 0, 127], [84, 84, 127], [84, 170, 127], [84, 255, 127],
    [170, 0, 127], [170, 84, 127], [170, 170, 127], [170, 255, 127], [255, 0, 127],
    [255, 84, 127], [255, 170, 127], [255, 255, 127], [0, 84, 255], [0, 170, 255],
    [0, 255, 255], [84, 0, 255], [84, 84, 255], [84, 170, 255], [84, 255, 255],
    [170, 0, 255], [170, 84, 255], [170, 170, 255], [170, 255, 255], [255, 0, 255],
    [255, 84, 255], [255, 170, 255], [42, 0, 0], [84, 0, 0], [127, 0, 0],
    [170, 0, 0], [212, 0, 0], [255, 0, 0], [0, 42, 0], [0, 84, 0],
    [0, 127, 0], [0, 170, 0], [0, 212, 0], [0, 255, 0], [0, 0, 42],
    [0, 0, 84], [0, 0, 127], [0, 0, 170], [0, 0, 212], [0, 0, 255],
    [0, 0, 0], [36, 36, 36], [72, 72, 72], [109, 109, 109], [145, 145, 145],
    [182, 182, 182], [218, 218, 218], [0, 113, 188], [80, 182, 188], [127, 127, 0]
]

def preprocess(img):
    height, width = img.shape[:2]
    img_resized = cv.resize(img, (image_size[1], image_size[0]))
    wpad = (image_size[0] + 31) // 32 * 32 - image_size[0]
    hpad = (image_size[1] + 31) // 32 * 32 - image_size[1]
    top, bottom = hpad // 2, hpad - hpad // 2
    left, right = wpad // 2, wpad - wpad // 2
    img_padded = cv.copyMakeBorder(img_resized, top, bottom, left, right, cv.BORDER_CONSTANT, value=0)

    in_mat = ncnn.Mat.from_pixels(img_padded, ncnn.Mat.PixelType.PIXEL_RGB2BGR, img_padded.shape[1], img_padded.shape[0])
    in_mat.substract_mean_normalize(mean_vals, norm_vals)
    return in_mat

def detect(net, img, prob_threshold=0.6):
    in_mat = preprocess(img)

    ex = net.create_extractor()
    ex.set_light_mode(False)
    ex.set_num_threads(4)
    ex.input("in0", in_mat)

    cls_pred_out = ncnn.Mat()
    ex.extract("out0", cls_pred_out)

    scores = np.array(cls_pred_out).flatten()
    label = np.argmax(scores)
    score = scores[label]

    objects = []
    if score >= prob_threshold and score < 1.0:
        obj = {
            'label': label,
            'prob': score
        }
        objects.append(obj)
        print(f"Detected object: label={class_names[label]}, prob={score:.2f}")

    return objects

def draw(img, objects):
    if not objects:
        print("No objects to draw!")
        return img

    obj = objects[0]  # 取第一个检测到的物体
    color = color_list[obj['label'] % num_classes]
    text = f"{class_names[obj['label']]} {obj['prob']*100:.1f}%"
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_size, baseline = cv.getTextSize(text, font, font_scale, thickness)

    x = img.shape[1] - text_size[0] - 10  # 右下角，距离右侧10像素
    y = img.shape[0] - 10  # 距离底部10像素
    if y < text_size[1] + baseline:
        y = text_size[1] + baseline
    if x < 0:
        x = 0

    cv.rectangle(img, (x, y - text_size[1] - baseline), (x + text_size[0], y + baseline), color, -1)
    text_color = (0, 0, 0) if sum(color) >= 381 else (255, 255, 255)
    cv.putText(img, text, (x, y), font, font_scale, text_color, thickness)

    return img

def load_ncnn_model(param_path, bin_path, use_gpu=False):
    try:
        net = ncnn.Net()
        has_gpu = ncnn.get_gpu_count() > 0
        if use_gpu and not has_gpu:
            print("Warning: GPU computation requested but no Vulkan-compatible GPU detected. Falling back to CPU.")
        net.opt.use_vulkan_compute = has_gpu and use_gpu
        net.opt.use_fp16_arithmetic = True
        net.opt.use_fp16_packed = True
        net.opt.use_fp16_storage = True
        net.load_param(param_path)
        net.load_model(bin_path)
        print(f"Using {'GPU' if net.opt.use_vulkan_compute else 'CPU'} for computation")
        return net
    except Exception as e:
        print(f"Error loading NCNN model: {e}")
        raise

def load_image(image_path):
    img = cv.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NCNN MobileNetV3 inference on an image.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()

    bin_path = "./params/mobilenetv3-TSC.bin"
    params_path = "./params/mobilenetv3-TSC.param"
    try:
        net = load_ncnn_model(params_path, bin_path, use_gpu=False)
        img = load_image(args.image_path)
        objects = detect(net, img, prob_threshold=0.6)
        img_with_text = draw(img, objects)
        cv.imwrite("output.jpg", img_with_text)
        print("Inference completed. Output saved as 'output.jpg'.")
    except Exception as e:
        print(f"Error during inference: {e}")
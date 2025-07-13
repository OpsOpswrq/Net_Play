import ncnn
import numpy as np
import cv2 as cv
import argparse
import math
import torch
import os

image_size = [320, 320]
reg_max = 7
num_classes = 58
strides = [8, 16, 32]
mean_vals = [103.53, 116.28, 123.675]
norm_vals = [1.0/57.375, 1.0/57.12, 1.0/58.395]

class_names = [
    "speed_five", "speed_fifteen", "speed_thirty", "speed_forty", "speed_fifty",
    "speed_sixty", "speed_seventy", "speed_eighty", "forbid_straight_left", "forbid_straight_right",
    "forbid_straight", "forbid_left", "forbid_left_right", "forbid_right", "forbid_overtake",
    "forbid_turn_around", "forbid_vehicle", "forbid_ring", "release_speed_limit_forty", "release_speed_limit_fifty",
    "straight_right", "straight", "left", "left_right", "right", "lean_left",
    "lean_right", "roundabout", "vehicle", "ring", "non_motor_vehicle",
    "turn_around", "detour_left_and_right", "notice_traffic_sign", "notice_danger", "pedestrian",
    "non_motor_vehicle_notice", "children", "sharp_right_turn", "sharp_left_turn", "downhill",
    "uphill", "slow_down", "right_T_crossroad", "left_T_crossroad", "village",
    "reverse_detour", "unattended_railway", "construction", "continuous_detour", "attended_railway",
    "accident_prone", "stop_sign", "no_through_traffic", "no_parking", "no_entry",
    "yield", "stop_for_inspection"
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

category = "TSC"

def softmax(input_data, size):
    if size <= 0:
        return np.zeros(size, dtype=np.float32)
    max_val = np.max(input_data)
    exp_data = np.exp(input_data - max_val)
    sum_exp = np.sum(exp_data)
    return exp_data / sum_exp

def intersection_area(rect1, rect2):
    x_left = max(rect1[0], rect2[0])
    y_top = max(rect1[1], rect2[1])
    x_right = min(rect1[0] + rect1[2], rect2[0] + rect2[2])
    y_bottom = min(rect1[1] + rect1[3], rect2[1] + rect2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    return (x_right - x_left) * (y_bottom - y_top)

def qsort_descent_inplace(faceobjects):
    if not faceobjects:
        return
    faceobjects.sort(key=lambda x: x['prob'], reverse=True)

def nms_sorted_bboxes(faceobjects, nms_threshold):
    picked = []
    areas = [obj['rect'][2] * obj['rect'][3] for obj in faceobjects]
    for i in range(len(faceobjects)):
        keep = True
        for j in picked:
            inter_area = intersection_area(faceobjects[i]['rect'], faceobjects[j]['rect'])
            union_area = areas[i] + areas[j] - inter_area
            if inter_area / union_area > nms_threshold:
                keep = False
                break
        if keep:
            picked.append(i)
    return picked

def generate_proposals(cls_pred, dis_pred, stride, in_pad_shape, prob_threshold):
    objects = []
    num_grid_y, num_grid_x, num_class = cls_pred.shape
    reg_max_1 = dis_pred.shape[2] // 4

    # print(f"generate_proposals: stride={stride}, num_grid_y={num_grid_y}, num_grid_x={num_grid_x}, num_class={num_class}, reg_max_1={reg_max_1}")

    sm = np.zeros(reg_max_1, dtype=np.float32)
    # num_grid_x = cls_pred.shape[1] - int(32 / stride)
    # print(f"Adjusted num_grid_x: {num_grid_x}")
    for i in range(num_grid_y):
        for j in range(num_grid_x):
            scores = cls_pred[i, j, :]
            label = np.argmax(scores)
            score = scores[label]

            if score >= prob_threshold and score < 1.0:
                pred_ltrb = np.zeros(4, dtype=np.float32)
                for k in range(4):
                    dis_values = dis_pred[i, j, k * reg_max_1:(k + 1) * reg_max_1]
                    sm = softmax(dis_values, reg_max_1)
                    dis = np.sum(np.arange(reg_max_1, dtype=np.float32) * sm)
                    pred_ltrb[k] = dis * stride # 看距离大小

                pb_cx = (j + 0.5) * stride
                pb_cy = (i + 0.5) * stride
                x0 = pb_cx - pred_ltrb[0]
                y0 = pb_cy - pred_ltrb[1]
                x1 = pb_cx + pred_ltrb[2]
                y1 = pb_cy + pred_ltrb[3] # 先进行softmax操作，获取最大概率的位置，之后在进行定位

                obj = {
                    'rect': [x0, y0, x1 - x0, y1 - y0],
                    'label': label,
                    'prob': score
                }
                objects.append(obj)
    # print(f"Generated proposals: {len(objects)}")
    return objects

def preprocess(img):
    height, width = img.shape[:2]
    w = width
    h = height
    scale = 1.0
    if w > h:
        scale = image_size[1] / w
        w = image_size[1]
        h = int(h * scale)
    else:
        scale = image_size[0] / h
        h = image_size[0]
        w = int(w * scale)

    img_resized = cv.resize(img, (w, h), interpolation=cv.INTER_LINEAR)
    wpad = (w + 31) // 32 * 32 - w
    hpad = (h + 31) // 32 * 32 - h
    top, bottom = hpad // 2, hpad - hpad // 2
    left, right = wpad // 2, wpad - wpad // 2
    img_padded = cv.copyMakeBorder(img_resized, top, bottom, left, right, cv.BORDER_CONSTANT, value=0)

    # print(f"Preprocess: original={width}x{height}, resized={w}x{h}, padded={img_padded.shape[1]}x{img_padded.shape[0]}, scale={scale}")

    in_mat = ncnn.Mat.from_pixels(img_padded, ncnn.Mat.PixelType.PIXEL_RGB2BGR, img_padded.shape[1], img_padded.shape[0])
    in_mat.substract_mean_normalize(mean_vals, norm_vals)
    return in_mat, scale, wpad, hpad

def detect(net, img, prob_threshold=0.6, nms_threshold=0.8):
    width, height = img.shape[1], img.shape[0]
    in_mat, scale, wpad, hpad = preprocess(img)

    # Define expected feature map sizes based on C++ output

    ex = net.create_extractor()
    # ex.set_light_mode(False)
    # ex.set_num_threads(4)
    ex.input("in0", in_mat)

    proposals = []

    # Stride 8
    dis_pred_out1 = ncnn.Mat()
    cls_pred_out1 = ncnn.Mat()
    ex.extract("159", dis_pred_out1)
    ex.extract("158", cls_pred_out1)
    cls_pred_np = np.array(cls_pred_out1).reshape(cls_pred_out1.w, cls_pred_out1.h, cls_pred_out1.c)
    dis_pred_np = np.array(dis_pred_out1).reshape(dis_pred_out1.w, dis_pred_out1.h, dis_pred_out1.c)
    # print(f"Stride 8: cls_pred shape={cls_pred_np.shape}, dis_pred shape={dis_pred_np.shape}")
    objects8 = generate_proposals(cls_pred_np, dis_pred_np, 8, in_mat.shape, prob_threshold)
    proposals.extend(objects8)

    # Stride 16
    dis_pred_out2 = ncnn.Mat()
    cls_pred_out2 = ncnn.Mat()
    ex.extract("175", dis_pred_out2)
    ex.extract("174", cls_pred_out2)
    cls_pred_np = np.array(cls_pred_out2).reshape(cls_pred_out2.w, cls_pred_out2.h, cls_pred_out2.c)
    dis_pred_np = np.array(dis_pred_out2).reshape(dis_pred_out2.w, dis_pred_out2.h, dis_pred_out2.c)
    # print(f"Stride 16: cls_pred shape={cls_pred_np.shape}, dis_pred shape={dis_pred_np.shape}")
    objects16 = generate_proposals(cls_pred_np, dis_pred_np, 16, in_mat.shape, prob_threshold)
    proposals.extend(objects16)

    # Stride 32
    dis_pred_out3 = ncnn.Mat()
    cls_pred_out3 = ncnn.Mat()
    ex.extract("191", dis_pred_out3)
    ex.extract("190", cls_pred_out3)
    cls_pred_np = np.array(cls_pred_out3).reshape(cls_pred_out3.w, cls_pred_out3.h, cls_pred_out3.c)
    dis_pred_np = np.array(dis_pred_out3).reshape(dis_pred_out3.w, dis_pred_out3.h, dis_pred_out3.c)
    # print(f"Stride 32: cls_pred shape={cls_pred_np.shape}, dis_pred shape={dis_pred_np.shape}")
    objects32 = generate_proposals(cls_pred_np, dis_pred_np, 32, in_mat.shape, prob_threshold)
    proposals.extend(objects32)

    # print(f"Total proposals before NMS: {len(proposals)}")

    qsort_descent_inplace(proposals)
    picked = nms_sorted_bboxes(proposals, nms_threshold)
    # print(f"Picked objects after NMS: {len(picked)}")

    objects = []
    for i in picked:
        obj = proposals[i]
        x0 = (obj['rect'][0] - wpad / 2) / scale
        y0 = (obj['rect'][1] - hpad / 2) / scale
        x1 = (obj['rect'][0] + obj['rect'][2] - wpad / 2) / scale
        y1 = (obj['rect'][1] + obj['rect'][3] - hpad / 2) / scale

        x0 = max(min(x0, width - 1), 0)
        y0 = max(min(y0, height - 1), 0)
        x1 = max(min(x1, width - 1), 0)
        y1 = max(min(y1, width - 1), 0)

        objects.append({
            'rect': [x0, y0, x1 - x0, y1 - y0],
            'label': obj['label'],
            'prob': obj['prob']
        })
        # print(f"Object {len(objects)-1}: x={x0:.2f}, y={y0:.2f}, w={(x1-x0):.2f}, h={(y1-y0):.2f}, prob={obj['prob']:.2f}, label={class_names[obj['label']]}")

    objects.sort(key=lambda x: x['rect'][2] * x['rect'][3], reverse=True)
    return objects

def draw(img, objects):
    if not objects:
        # print("No objects to draw!")
        return img

    color_index = 0
    for obj in objects:
        color = color_list[color_index % num_classes]
        color_index += 1
        x, y, w, h = [int(v) for v in obj['rect']]
        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
        # 确定那些图像成功了
        if int(obj['label']) == int(category):
            print(class_names[obj['label']], "is verified")
        text = f"{class_names[obj['label']]} {obj['prob']*100:.1f}%"
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size, baseline = cv.getTextSize(text, font, font_scale, thickness)

        tx = x
        ty = y - text_size[1] - baseline
        if ty < 0:
            ty = 0
        if tx + text_size[0] > img.shape[1]:
            tx = img.shape[1] - text_size[0]

        cv.rectangle(img, (tx, ty), (tx + text_size[0], ty + text_size[1] + baseline), color, -1)
        text_color = (0, 0, 0) if sum(color) >= 381 else (255, 255, 255)
        cv.putText(img, text, (tx, ty + text_size[1]), font, font_scale, text_color, thickness)

    return img

def load_ncnn_model(param_path, bin_path, use_gpu=False):
    try:
        net = ncnn.Net()
        has_gpu = ncnn.get_gpu_count() > 0
        if use_gpu and not has_gpu:
            print("Warning: GPU computation requested but no Vulkan-compatible GPU detected. Falling back to CPU.")
        net.opt.use_vulkan_compute = has_gpu and use_gpu
        net.opt.num_threads = 4
        net.opt.use_fp16_arithmetic = True
        net.opt.use_fp16_packed = True
        net.opt.use_fp16_storage = True
        net.load_param(param_path)
        net.load_model(bin_path)
        # print(f"Using {'GPU' if net.opt.use_vulkan_compute else 'CPU'} for computation")
        return net
    except Exception as e:
        print(f"Error loading NCNN model: {e}")
        raise

def load_image(image_path):
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    # img_resized = cv.resize(img, (640, 576), interpolation = cv.INTER_AREA) # 640 x 576 is the expected input size
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    return img

# 这个的处理逻辑和C++还是不太一样，因为cls和dis的shape和C++的shape不一样，这导致模型出现问题
# 感觉还是模型不太行，还是需要进行微调操作，C++和Python对模型的处理逻辑不太一样
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NCNN NanoDet inference on an image with NVIDIA GeForce RTX 4060 Ti.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()

    bin_path = "./params/nanodet-TSC.bin"
    param_path = "./params/nanodet-TSC.param"
    try:
        if os.path.isdir(args.image_path):
            for image_dir in os.listdir(args.image_path):
                child_dir_path = os.path.join(args.image_path, image_dir)
                if os.path.isdir(child_dir_path):
                    for image_file in os.listdir(child_dir_path):
                        category = image_dir
                        if image_file.endswith(('.jpg', '.png', '.jpeg')):
                            image_path = os.path.join(child_dir_path, image_file)
                            net = load_ncnn_model(param_path, bin_path, use_gpu=True)
                            img = load_image(image_path)
                            objects = detect(net, img, prob_threshold=0.6, nms_threshold=0.5)
                            img_with_boxes = draw(img, objects)
                            cv.imwrite("output.jpg", img_with_boxes)
                            # print(f"Inference completed for {image_file}. Output saved as '{output_path}'.")
                else:
                    # 需要进行拼接处理
                    category = args.image_path.split("/")[1]
                    if image_dir.endswith(('.jpg', '.png', '.jpeg')):
                        net = load_ncnn_model(param_path, bin_path, use_gpu=True)
                        img = load_image(os.path.join(args.image_path, image_dir))
                        objects = detect(net, img, prob_threshold=0.6, nms_threshold=0.5)
                        img_with_boxes = draw(img, objects)
                        cv.imwrite("output.jpg", img_with_boxes)
                        # print(f"Inference completed for {image_dir}. Output saved as '{output_path}'.")
        else:
            category = args.image_path.split("/")[2] # 直接获得类别名字
            net = load_ncnn_model(param_path, bin_path, use_gpu=True)
            img = load_image(args.image_path)
            objects = detect(net, img, prob_threshold=0.6, nms_threshold=0.5)
            img_with_boxes = draw(img, objects)
            cv.imwrite("output.jpg", img_with_boxes)
            # print("Inference completed. Output saved as 'output.jpg'.")
    except Exception as e:
        print(f"Error during inference: {e}")
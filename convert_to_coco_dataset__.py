import os
import json
from PIL import Image

# 定义图片文件夹和标签文件夹的路径
image_dir = '../dataset/images'
label_dir = '../dataset/labels'

# 创建COCO数据集的基本结构
coco_data = {
    "info": {},
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": []
}

# 添加分类信息，例如：如果你的数据集是用于检测任务，可以添加"person"和"car"等类别
categories = [
        "BPSK",
        "BPSK_X",
        "QPSK",
        "OQPSK",
        "8-PSK",
        "QAM16",
        "QAM32",
        "QAM64",
        "QAM128",
        "QAM256", #10-1
        "QAM512",
        # "QAM1024",
        # "QAM2048",
        # "QAM4096",
        "APSK16",
        "APSK32",
        "PAM4",
        "CPM",
        "QAM8",
        "OOK",
        "none",
        "increasing",
        "decreasing", #20-1
        "v-shape",
        "inverted v",
        "barker-2 +-",
        "barker-2 ++",
        "barker-3",
        "barker-4 ++-+",
        "barker-4 +++-",
        "barker-5",
        "barker-7",
        "barker-11", #30-1
        "barker-13",
        "frank-4",
        "frank-6",
        # "fmcw",
        "2FSK",
        "costas",
        "frequency_hop",
        "nlfm_sine",
        # "nlfm_atan", ## 还没整，但是可以搞
        "adsb",
        ]  # replace with your categories
for i, category in enumerate(categories):
    coco_data["categories"].append({"id": i+1, "name": category, "supercategory": "object"})
# coco_data["categories"].append({"id": 1, "name": "person", "supercategory": "object"})
# coco_data["categories"].append({"id": 2, "name": "car", "supercategory": "object"})

# 读取图片文件夹中的所有图片文件
image_files = os.listdir(image_dir)

# 遍历每张图片
for img_id, image_file in enumerate(image_files):
    image_path = os.path.join(image_dir, image_file)

    # 获取图片信息
    img = Image.open(image_path)
    width, height = img.size

    # 添加图片信息到COCO数据集中
    coco_data["images"].append({"id": img_id, "file_name": image_file, "width": width, "height": height})

    # 读取对应的标签文件
    label_file = os.path.join(label_dir, os.path.splitext(image_file)[0] + ".txt")
    with open(label_file, 'r') as f:
        lines = f.readlines()

    # 遍历每行标注信息
    for line in lines:
        # 解析每行标注信息，例如：如果标注信息格式为"x_min, y_min, x_max, y_max, category_id"，可以使用如下代码
        parts = line.strip().split()
        # x_min, y_min, x_max, y_max, category_id = map(float, parts)
        category_id, x_min, x_max, bw, center, snr = map(float, parts)
        y_min = center-0.5*bw
        y_max = center+0.5*bw
        x_min, x_max, y_min, y_max = x_min*width, x_max*width, y_min*height, y_max*height
        # 计算bbox和segmentation
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        segmentation = [[x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min]]

        # 添加标注信息到COCO数据集中
        coco_data["annotations"].append({
            "id": len(coco_data["annotations"]) + 1,
            "image_id": img_id,
            "category_id": int(category_id),
            "segmentation": segmentation,
            "area": (x_max - x_min) * (y_max - y_min),
            "bbox": bbox,
            "iscrowd": 0,
            'center': center,
            'snr': snr
        })

# 将COCO数据集保存为JSON文件
with open('dataset.json', 'w') as f:
    json.dump(coco_data, f, indent=4)
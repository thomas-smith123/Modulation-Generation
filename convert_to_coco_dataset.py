import os
import json
import cv2
import random,shutil
import numpy as np
from pycocotools import mask
from skimage import measure

# Define the directory of your images and labels
img_dir = '../m_dataset/images'
label_dir = '../m_dataset/labels'

# Initialize the COCO dataset dictionary
coco_dataset = {
    "images": [],
    "annotations": [],
    "categories": []
}

# Define your categories
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
    "QAM256",
    "QAM512",
    "APSK16",
    "APSK32",
    "PAM4",
    "QAM8",
    "OOK",
    "none",
    "increasing",
    "decreasing",
    "v-shape",
    "inverted v",
    "barker-2 +-",
    "barker-2 ++",
    "barker-3",
    "barker-4 ++-+",
    "barker-4 +++-",
    "barker-5",
    "barker-7",
    "barker-11",
    "barker-13",
    "frank-4",
    "frank-6",
    "2FSK",
    "costas",
    "frequency_hop",
    "nlfm_sine",
    "fmcw",
    "adsb",
    "dsb"
        ]  # replace with your categories
all_images = os.listdir(img_dir)

# Randomly split the images into training and testing sets
random.shuffle(all_images)
split_index = int(len(all_images) * 0.8)  # 80% for training, 20% for testing
train_images = all_images[:split_index]
test_images = all_images[split_index:]

# Create the COCO dataset for each set
for image_set, images in [('train', train_images), ('test', test_images)]:
    coco_dataset = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    for i, category in enumerate(categories):
        coco_dataset['categories'].append({
            'id': i + 1,
            'name': category,
            'supercategory': 'none'
        })

    annotation_id = 1
    for i, img_name in enumerate(images):
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, img_name.replace('.png', '.txt'))  # replace with your label file extension

        # Get image details
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        # Add image information to the dataset
        coco_dataset['images'].append({
            'id': i + 1,
            'file_name': img_name,
            'width': width,
            'height': height
        })

        # Read labels and add annotation information to the dataset
        with open(label_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                # category_index, x_center, y_center, w, h, snr = map(float, line.strip().split())
                # category_index = int(category_index)
                #
                # x_center *= width
                # y_center *= height
                # w *= width
                # h *= height
                #
                # x = x_center - w / 2
                # y = y_center - h / 2
                category_index, x, end, bw, cent, snr = map(float, line.strip().split())
                category_index = int(category_index)
                y=0 if cent-0.5*bw<0 else cent-0.5*bw
                w=end-x
                h=bw

                x *= width
                y *= height
                w *= width
                h *= height

                # Generate the segmentation polygon
                mask = np.zeros((height, width), np.uint8)
                cv2.rectangle(mask, (int(x), int(y)), (int(x + w), int(y + h)), 255, -1)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                segmentation = []
                for contour in contours:
                    contour = contour.flatten().tolist()
                    if len(contour) > 4:
                        segmentation.append(contour)


                coco_dataset['annotations'].append({
                    'id': annotation_id,
                    'image_id': i + 1,
                    'category_id': category_index + 1,
                    'bbox': [x, y, w, h],
                    'area': w * h,
                    'iscrowd': 0,
                    'segmentation': segmentation,
                    'snr': snr,
                    'center': cent
                })

                annotation_id += 1

    # Save the COCO dataset as a JSON file
    with open(f'coco_{image_set}.json', 'w') as file:
        json.dump(coco_dataset, file, indent=4)


import json

def yolo2coco(addr):
    train_path = addr+'/train.txt'
    valid_path = addr+'/valid.txt'
    test_paht = addr+'/test.txt'
    __yolo2coco__(train_path)
def convert_yolo_to_coco(x_center, y_center, width, height, img_width=512, img_height=512):
    x_min = (x_center - width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    width = width * img_width
    height = height * img_height
    return [x_min, y_min, width, height]

def __yolo2coco__(filepath):
    categories = [
    {"id": 1, "name": "category1"},
    {"id": 2, "name": "category2"},
    # 添加更多类别
]
    annotation_id = 1
    struc = {
        "images": [],
        "annotations": [],
        "categories": categories}

    label_path = filepath.split('/')[0]
    with open(filepath,'r') as f:
        ff = open(filepath.replace('txt','json'),'w')
        for line in f:
            images = line.split('/')[-1].split('\n')[0] ## 不确定后面的\n有没有用
            label_file = open(label_path+line.split('\n')[0].split('.')[-2].replace('images','labels')+'.txt','r')
            image_info = {
                "file_name": images,
                "id": len(struc["images"]) + 1,
                "width": 512,
                "height": 512
            }
            struc["images"].append(image_info)
            
            for i in label_file:
                category_id, x_center, y_center, width, height = map(float, i.split())
                bbox = convert_yolo_to_coco(x_center, y_center, width, height)
                annotation = {
                            "id": annotation_id,
                            "image_id": image_info["id"],
                            "category_id": int(category_id) + 1,
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],
                            "iscrowd": 0
                        }
                struc["annotations"].append(annotation)
                annotation_id += 1
        json.dump(struc, ff,indent=4)
        ff.close()
    

if __name__ == '__main__':
    yolo2coco('DataSet2')
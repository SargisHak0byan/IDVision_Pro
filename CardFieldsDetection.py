import torch
import numpy as np
import os
import cv2
import pandas as pd
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize

from data_aug.data_aug import *

IMAGE_SIZE = 512



class CardFieldsDetection(Dataset):
    def __init__(self, data_path, csv_path, is_Train=True, transform=None):
        self.data = []
        self.annotations = []
        self.obj_labels = []
        self.masks = []
        self.transform = transform

        # defining directory to store augmented images
        class_name = data_path.split("/")[-1]
        aug_root = "./Dataset_Annotations"
        aug_path = os.path.join(aug_root, class_name)
        aug_image_id = 102

        # open annotation files
        labels_1_50 = pd.read_csv(os.path.join(csv_path, "labeled_1_50.csv"))
        labels_51_101 = pd.read_csv(os.path.join(csv_path, "labeled_51_101.csv"))
        labels_102_125 = pd.read_csv(os.path.join(csv_path, "new_brit_ID_ant.csv"))

        # get bbox values
        labels_1_50['region_shape_attributes'] = labels_1_50['region_shape_attributes'].apply(lambda x: json.loads(x))
        labels_51_101['region_shape_attributes'] = labels_51_101['region_shape_attributes'].apply(
            lambda x: json.loads(x))
        labels_102_125['region_shape_attributes'] = labels_102_125['region_shape_attributes'].apply(
            lambda x: json.loads(x))

        for filename in os.listdir(data_path):
            image_id = int(filename.split(".")[0])
            image_path = os.path.join(data_path, filename)
            region_attributes = None
            if image_id <= 125:
                # find annotation of original images in the csv
                if image_id < 51:
                    region_attributes = labels_1_50[labels_1_50.filename == filename]['region_shape_attributes']
                elif 51 <= image_id <= 101:
                    region_attributes = labels_51_101[labels_51_101.filename == filename]['region_shape_attributes']
                if not is_Train and 101 <= image_id <= 125:
                    region_attributes = labels_102_125[labels_102_125.filename == filename]['region_shape_attributes']

                    # read image
                img = cv2.imread(image_path)[:, :, ::-1]  # OpenCV uses BGR channels
                height, width, channels = img.shape

                # convert annotations to xmin xman ymin ymax format
                annotation = []
                for idx, item in enumerate(region_attributes):
                    x1 = item['x']
                    x2 = x1 + item['width']

                    y1 = item['y']
                    y2 = y1 + item['height']

                    annotation.append([x1, y1, x2, y2, idx + 1])

                    # skip unannotated images
                if len(annotation) == 0:
                    continue

                # resize image and its bboxes
                resizer = Resize(IMAGE_SIZE)
                img, annotation = resizer(img, np.array(annotation, dtype='float'))
                path = aug_path + f"/{image_id}.jpg" if is_Train else aug_root + f"/val/{image_id}.jpg"
                cv2.imwrite(path, img)

                # save required information
                self.data.append(path)
                self.annotations.append([x[:-1] for x in annotation])
                self.obj_labels.append([x[-1] for x in annotation])

                if is_Train:
                    # define augmentations from external library
                    transforms = [
                        RandomHorizontalFlip(1),
                        RandomScale(0.3, diff=True),
                        RandomTranslate(0.3, diff=True),
                        RandomRotate(30), RandomRotate(60), RandomRotate(90), RandomRotate(120),
                        RandomRotate(180),
                        RandomShear(0.2),
                        RandomHSV(100, 100, 100),
                    ]

                    # augment image and its bboxes accordingly
                    for transform in transforms:
                        aug_image_path = aug_path + f"/{aug_image_id}.jpg"
                        aug_image_id += 1

                        annotation = np.array(annotation, dtype='float')
                        aug_img, aug_annotation = transform(img.copy(), annotation.copy())
                        try:
                            cv2.imwrite(aug_image_path, aug_img)  # save augmented image
                            # save required information
                            self.data.append(aug_image_path)
                            self.annotations.append([x[:-1] for x in aug_annotation])
                            self.obj_labels.append([x[-1] for x in aug_annotation])
                        except:
                            continue

        print(f"[Dataset/__init__] {len(self.data)} images.")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path, annotation, obj_labels = self.data[idx], self.annotations[idx], self.obj_labels[idx]
        N = len(obj_labels)

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        # define requirements for input to model
        image_id = torch.tensor([idx])
        labels = torch.as_tensor(obj_labels, dtype=torch.int64)
        boxes = torch.as_tensor(annotation, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((N,), dtype=torch.int64)
        masks = torch.zeros((N, h, w), dtype=torch.uint8)

        for i, box in enumerate(np.array(annotation, dtype='int32')):
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[i, row_s:row_e, col_s:col_e] = 1

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target

    def __len__(self):
        return len(self.data)

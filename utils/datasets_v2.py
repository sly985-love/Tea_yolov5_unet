import glob
import os

import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from utils.utils import draw_umich_gaussian
import json



class KeyPointDatasets(Dataset):
    def __init__(self, root_dir, transforms=None):
        super(KeyPointDatasets, self).__init__()

        self.resize_W = 224
        self.resize_H = 224

        self.down_ratio = 1
        self.img_w = self.resize_W // self.down_ratio
        self.img_h = self.resize_H // self.down_ratio
        self.data_path = root_dir
        # self.img_path = os.path.join(root_dir, "images")
        self.image_files_path, self.json_files_path = self.read_data_set()

        # self.img_list = glob.glob(os.path.join(self.img_path, "*.jpg"))
        # self.txt_list = [item.replace(".jpg", ".txt").replace(
        #     "images", "labels") for item in self.img_list]

        if transforms is not None:
            self.transforms = transforms

    def read_data_set(self):
        all_img_files = []
        all_json_files = []

        img_list = os.listdir(self.data_path + "/" + "images")
        for img in img_list:
            img_abs_path = self.data_path + "/images/{}".format(img)
            houzui = img.split(".")[1]
            if houzui == "jpg":
                json_name = img.replace("jpg", "json")
            else:
                json_name = img.replace("png", "json")
            json_abs_name = self.data_path + "/jsons/{}".format(json_name)
            all_img_files.append(img_abs_path)
            all_json_files.append(json_abs_name)

        return all_img_files, all_json_files




    def __getitem__(self, index):
        img = self.image_files_path[index]
        json_file = self.json_files_path[index]
        # print(img)
        # print("----------------")
        try:

            img = cv2.imread(img)
            H, W = img.shape[:2]

            if self.transforms:
                img = self.transforms(img)

            # print("======================")

            # label = []
            #
            # with open(txt, "r") as f:
            #     for i, line in enumerate(f):
            #         if i == 0:
            #             # 第一行
            #             num_point = int(line.strip())
            #         else:
            #             x1, y1 = [(t.strip()) for t in line.split()]
            #             # range from 0 to 1
            #             x1, y1 = float(x1), float(y1)
            #
            #             cx, cy = x1 * self.img_w, y1 * self.img_h
            #
            #             heatmap = np.zeros((self.img_h, self.img_w))
            #
            #             draw_umich_gaussian(heatmap, (cx, cy), 30)

            gt = []
            with open(json_file, 'r') as f:
                gt_json = json.load(f)
                gt.append(gt_json["shapes"][0]["points"][0])
                gt.append(gt_json["shapes"][1]["points"][0])
                gt.append(gt_json["shapes"][2]["points"][0])
                gt.append(gt_json["shapes"][3]["points"][0])

            # print("+++++++++++++++++++++")


            heatmap_0 = np.zeros((self.img_h, self.img_w))
            draw_umich_gaussian(heatmap_0, (gt[0][0] * (self.resize_W/W), gt[0][1] * (self.resize_H/H)), 30)
            heatmap_1 = np.zeros((self.img_h, self.img_w))
            draw_umich_gaussian(heatmap_1, (gt[1][0] * (self.resize_W/W), gt[1][1] * (self.resize_H/H)), 30)
            heatmap_2 = np.zeros((self.img_h, self.img_w))
            draw_umich_gaussian(heatmap_2, (gt[2][0] * (self.resize_W/W), gt[2][1] * (self.resize_H/H)), 30)
            heatmap_3 = np.zeros((self.img_h, self.img_w))
            draw_umich_gaussian(heatmap_3, (gt[3][0] * (self.resize_W/W), gt[3][1] * (self.resize_H/H)), 30)

            heatmap = np.array([heatmap_0, heatmap_1, heatmap_2, heatmap_3])

            return img, torch.tensor(heatmap)

        except Exception as Error:
            print(Error)



    def __len__(self):
        return len(self.image_files_path)

    @staticmethod
    def collect_fn(batch):
        imgs, labels = zip(*batch)
        return torch.stack(imgs, 0), torch.stack(labels, 0)


class KeyPointDatasets_6P(Dataset):
    def __init__(self, root_dir, transforms=None):
        super(KeyPointDatasets_6P, self).__init__()

        self.resize_W = 256
        self.resize_H = 256

        self.down_ratio = 1
        self.img_w = self.resize_W // self.down_ratio
        self.img_h = self.resize_H // self.down_ratio
        self.data_path = root_dir
        # self.img_path = os.path.join(root_dir, "images")
        self.image_files_path, self.json_files_path = self.read_data_set()

        # self.img_list = glob.glob(os.path.join(self.img_path, "*.jpg"))
        # self.txt_list = [item.replace(".jpg", ".txt").replace(
        #     "images", "labels") for item in self.img_list]

        if transforms is not None:
            self.transforms = transforms

    def read_data_set(self):
        all_img_files = []
        all_json_files = []

        img_list = os.listdir(self.data_path + "/" + "images")
        for img in img_list:
            img_abs_path = self.data_path + "/images/{}".format(img)
            houzui = img.split(".")[1]
            if houzui == "jpg":
                json_name = img.replace("jpg", "json")
            else:
                json_name = img.replace("png", "json")
            json_abs_name = self.data_path + "/jsons/{}".format(json_name)
            all_img_files.append(img_abs_path)
            all_json_files.append(json_abs_name)

        return all_img_files, all_json_files

    def __getitem__(self, index):
        img = self.image_files_path[index]
        json_file = self.json_files_path[index]
        try:
            # print(img)
            img = cv2.imread(img)
            H, W = img.shape[:2]

            if self.transforms:
                img = self.transforms(img)

            gt = []
            with open(json_file, 'r') as f:
                gt_json = json.load(f)
                gt.append(gt_json["shapes"][0]["points"][0])
                gt.append(gt_json["shapes"][1]["points"][0])
                gt.append(gt_json["shapes"][2]["points"][0])
                gt.append(gt_json["shapes"][3]["points"][0])
                gt.append(gt_json["shapes"][4]["points"][0])
                gt.append(gt_json["shapes"][5]["points"][0])

            heatmap_0 = np.zeros((self.img_h, self.img_w))
            draw_umich_gaussian(heatmap_0, (gt[0][0] * (self.resize_W/W), gt[0][1] * (self.resize_H/H)), 30)
            heatmap_1 = np.zeros((self.img_h, self.img_w))
            draw_umich_gaussian(heatmap_1, (gt[1][0] * (self.resize_W/W), gt[1][1] * (self.resize_H/H)), 30)
            heatmap_2 = np.zeros((self.img_h, self.img_w))
            draw_umich_gaussian(heatmap_2, (gt[2][0] * (self.resize_W/W), gt[2][1] * (self.resize_H/H)), 30)
            heatmap_3 = np.zeros((self.img_h, self.img_w))
            draw_umich_gaussian(heatmap_3, (gt[3][0] * (self.resize_W/W), gt[3][1] * (self.resize_H/H)), 30)
            heatmap_4 = np.zeros((self.img_h, self.img_w))
            draw_umich_gaussian(heatmap_4, (gt[4][0] * (self.resize_W / W), gt[4][1] * (self.resize_H / H)), 30)
            heatmap_5 = np.zeros((self.img_h, self.img_w))
            draw_umich_gaussian(heatmap_5, (gt[5][0] * (self.resize_W / W), gt[5][1] * (self.resize_H / H)), 30)

            heatmap = np.array([heatmap_0, heatmap_1, heatmap_2, heatmap_3, heatmap_4, heatmap_5])

            return img, torch.tensor(heatmap)

        except Exception as Error:
            print(Error)

    def __len__(self):
        return len(self.image_files_path)

    @staticmethod
    def collect_fn(batch):
        imgs, labels = zip(*batch)
        return torch.stack(imgs, 0), torch.stack(labels, 0)



if __name__ == "__main__":
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((360, 480)),
        transforms.ToTensor(),
    ])
    kp_datasets = KeyPointDatasets(
        root_dir=r"D:\GraceKafuu\CJXM\28.paper_keypoint_detection\data", transforms=trans)

    # for i in range(len(kp_datasets)):
    # print(kp_datasets[i][0].shape, kp_datasets[i][1])

    data_loader = DataLoader(kp_datasets, num_workers=0, batch_size=4, shuffle=True,
                             collate_fn=kp_datasets.collect_fn
                             )

    for data, label in data_loader:
        print(data.shape, label.shape)

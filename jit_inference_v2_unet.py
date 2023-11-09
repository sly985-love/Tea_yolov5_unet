#coding=utf-8

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
# import pandas as pd
from utils.datasets_v2 import KeyPointDatasets_6P
# from model import KeyPointModel
# from models import KFSG
import cv2
import time
from PIL import Image
from torchvision import transforms
import argparse
import os


def toTensor(img):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    #img = torch.from_numpy(img)
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256


def get_peak_points(heatmaps):
    """

    :param heatmaps: numpy array (N,4,256,256)
    :return:numpy array (N,4,2) #
    """
    N,C,H,W = heatmaps.shape   # N= batch size C=4 hotmaps
    all_peak_points = []
    for i in range(N):
        peak_points = []
        for j in range(C):
            yy,xx = np.where(heatmaps[i, j] == heatmaps[i, j].max())
            y = yy[0]
            x = xx[0]
            peak_points.append([x, y])
        all_peak_points.append(peak_points)
    all_peak_points = np.array(all_peak_points)
    return all_peak_points



def inference(image_path):
    base_name = os.path.basename(image_path)
    parser = argparse.ArgumentParser(description="model path")
    parser.add_argument('--model', type=str, default="weights/U_net_460.pt")

    args = parser.parse_args()
    SIZE = 256, 256

    transforms_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    # model = KeyPointModel()
    # model = KFSG.KFSGNet()
    # model.eval()
    # model.load_state_dict(torch.load(args.model))

    model = torch.jit.load(args.model)
    model.eval()

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]
    img_tensor = transforms_test(img)
    img_tensor = img_tensor * 255
    img_tensor = img_tensor.unsqueeze(dim=0).to("cuda:0")
    pred_heatmaps = model(img_tensor)
    # print(pred_heatmaps.shape)

    # pil_img = Image.open(image_path)
    # pil_img = pil_img.resize((256, 256), Image.ANTIALIAS)
    # pil_img = np.array(pil_img)
    # x = pil_img.reshape((3, 256, 256)).astype(np.float32)
    # x = x / 255.
    # x_tensor = torch.from_numpy(x).view(1, 3, 256, 256).to("cuda:0")
    # pred_heatmaps = net(x_tensor)

    # # image = image[0:height, 0:height]
    # image_resized = cv2.resize(image, (128, 128))
    # image1 = Variable(toTensor(image_resized)).cuda()
    # pred_heatmaps = net(image1)


    # cv2.imshow("heatmap0", pred_heatmaps.cpu().data.numpy()[0][0])
    # cv2.imshow("heatmap1", pred_heatmaps.cpu().data.numpy()[0][1])
    # cv2.imshow("heatmap2", pred_heatmaps.cpu().data.numpy()[0][2])
    # cv2.imshow("heatmap3", pred_heatmaps.cpu().data.numpy()[0][3])

    pred_points = get_peak_points(pred_heatmaps.cpu().data.numpy())  # (N,4,2)
    pred_points = pred_points.reshape((pred_points.shape[0], -1))  # (N,8)
    # print("pred_points: ", pred_points)

    point_0 = [int(round(pred_points[0][0] * (W / 256))), int(round(pred_points[0][1] * (H / 256)))]
    point_1 = [int(round(pred_points[0][2] * (W / 256))), int(round(pred_points[0][3] * (H / 256)))]
    point_2 = [int(round(pred_points[0][4] * (W / 256))), int(round(pred_points[0][5] * (H / 256)))]
    point_3 = [int(round(pred_points[0][6] * (W / 256))), int(round(pred_points[0][7] * (H / 256)))]
    point_4 = [int(round(pred_points[0][8] * (W / 256))), int(round(pred_points[0][9] * (H / 256)))]
    point_5 = [int(round(pred_points[0][10] * (W / 256))), int(round(pred_points[0][11] * (H / 256)))]


    cv2.circle(img, (point_0[0], point_0[1]), 2, (255, 0, 255), 2)
    cv2.circle(img, (point_1[0], point_1[1]), 2, (255, 0, 255), 2)
    cv2.circle(img, (point_2[0], point_2[1]), 2, (255, 0, 255), 2)
    cv2.circle(img, (point_3[0], point_3[1]), 2, (255, 0, 255), 2)
    cv2.circle(img, (point_4[0], point_4[1]), 2, (255, 0, 255), 2)
    cv2.circle(img, (point_5[0], point_5[1]), 2, (255, 0, 255), 2)

    image_resized = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/{}".format(base_name), image_resized)

    # cv2.imshow("result", image_resized)
    # cv2.waitKey(0)




if __name__ == '__main__':

    # inference("data/images/1.jpg")
    base_path = r"data/crops_v2/images".replace("\\", "/")

    img_list = os.listdir(base_path)
    for img in img_list:
        img_rela_path = "{}/{}".format(base_path, img)
        if img_rela_path.endswith(".jpg"):
            inference(img_rela_path)

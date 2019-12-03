import numpy as np
import cv2, json, torch
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
main_path = "/home/ubuntu/exam_data/train/"

RESIZE_TO = 50
def crop_segment(img, json_attribute):
    # im = cv2.imread(path + '.png')
    x,y = json_attribute["bounding_box"]["minimum"]["r"],json_attribute["bounding_box"]["minimum"]["c"]
    x2,y2 = json_attribute["bounding_box"]["maximum"]["r"],json_attribute["bounding_box"]["maximum"]["c"]
    return np.array(cv2.resize(img[x:x2,y:y2], (RESIZE_TO, RESIZE_TO)))
#     cv2.imwrite("roi3.jpg", roi)
train_data = []
train_labels = []
for j in range(0,4): #1328)
    try:
        img_path = main_path + "cells_" + str(j) + ".png"
        json_path = main_path + "cells_" + str(j) + ".json"
        im = cv2.imread(img_path)
        with open(json_path) as f:
            d = json.load(f)
        for i in range(len(d)):
            train_data.append([crop_segment(im, d[i]), d[i]["category"]])
#             train_data.append([crop_segment(im, d[i])])
#             train_labels.append(d[i]["category"])
    except FileNotFoundError:
        print("FileNotFoundError",img_path)
print(train_data[0])
# trainloader = torch.utils.data.DataLoader(train_data, shuffle=True)

import scipy.io as io
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random

all_images = sorted(os.listdir('./data/lsp_dataset/images'))

joints_info = np.load('joints_information.npy',allow_pickle=True)
print(joints_info.shape)
x_cords_read = joints_info[0]
y_cords_read = joints_info[1]
visible_read = joints_info[2]

x_cords_aug = []
y_cords_aug = []

for i in range(1800):
    im = cv2.imread('./data/lsp_dataset/images/'+all_images[i])
    im = np.rot90(im,2)
    h,w,c = im.shape
    x_aug = x_cords_read[i] * -1
    x_aug = x_aug + w
    y_aug = y_cords_read[i] * -1
    y_aug = y_aug + h
    x_cords_aug.append(x_aug)
    y_cords_aug.append(y_aug)

x_cords_aug = np.array(x_cords_aug)
y_cords_aug = np.array(y_cords_aug)
joints_info_aug = np.array([x_cords_aug,y_cords_aug,visible_read[:1800]])
print(joints_info_aug.shape)

np.save('joints_info_aug',joints_info_aug)



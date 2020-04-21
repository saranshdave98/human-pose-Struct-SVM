import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

size = 10
all_images = sorted(os.listdir('./data/lsp_dataset/images'))
test_images = all_images[1800:1800+size]

predicted_joints_x = np.load('StructSVM_predictions_50_10.npy',allow_pickle=10)[:,:14]
predicted_joints_y = np.load('StructSVM_predictions_50_10.npy',allow_pickle=10)[:,14:]
print(predicted_joints_x.shape)
print(predicted_joints_y.shape)
# predicted_joints_x = np.load('predicted_loc_x_aug_inc.npy',allow_pickle=True).T
# predicted_joints_y = np.load('predicted_loc_y_aug_inc.npy',allow_pickle=True).T

for i in range(size):
    Xs = predicted_joints_x[i]
    Ys = predicted_joints_y[i]
    im = cv2.imread('./data/lsp_dataset/images/'+test_images[i])
    plt.scatter(Xs,Ys)
    plt.imshow(im)
    plt.show()
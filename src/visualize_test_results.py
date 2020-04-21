import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

size = 10
all_images = sorted(os.listdir('./data/lsp_dataset/images'))
test_images = all_images[:size]

predicted_joints_x = np.load('StructSVM_predictions_150_10.npy',allow_pickle=10)[:,:14]
predicted_joints_y = np.load('StructSVM_predictions_150_10.npy',allow_pickle=10)[:,14:]
print(predicted_joints_x.shape)
print(predicted_joints_y.shape)
# predicted_joints_x = np.load('predicted_loc_x_aug_inc.npy',allow_pickle=True).T
# predicted_joints_y = np.load('predicted_loc_y_aug_inc.npy',allow_pickle=True).T

for i in range(size):
    Xs = predicted_joints_x[i]
    Ys = predicted_joints_y[i]
    im = cv2.imread('./data/lsp_dataset/images/'+test_images[i])
  
    #skeleton
    plt.scatter(Xs,Ys,color='black',marker='o')
    plt.plot(Xs[:2],Ys[:2])
    plt.plot(Xs[1:3],Ys[1:3])
    plt.plot(Xs[2:4],Ys[2:4])
    plt.plot(Xs[3:5],Ys[3:5])
    plt.plot(Xs[4:6],Ys[4:6])
    plt.plot(Xs[6:8],Ys[6:8])
    plt.plot(Xs[7:9],Ys[7:9])
    plt.plot(Xs[9:11],Ys[9:11])
    plt.plot(Xs[10:12],Ys[10:12])
    plt.plot(Xs[12:14],Ys[12:14])
    plt.plot([Xs[8],Xs[12]],[Ys[8],Ys[12]])
    plt.plot([Xs[9],Xs[12]],[Ys[9],Ys[12]])
    plt.plot([Xs[2],Xs[12]],[Ys[2],Ys[12]])
    plt.plot([Xs[3],Xs[12]],[Ys[3],Ys[12]])
    plt.imshow(im)
    plt.show()
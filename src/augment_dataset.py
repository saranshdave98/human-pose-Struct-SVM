import scipy.io as io
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random
import sklearn.cluster as cluster
import sys
import matplotlib.pyplot as plt

codebook_words = 400

#reading joints info
joints_info = np.load('joints_information.npy',allow_pickle=True)

#train test split
indices = np.arange(0,2000)
train_index = indices[:1800]
test_index = indices[1800:]

all_images = sorted(os.listdir('./data/lsp_dataset/images'))

train_images = []
for index in train_index:
    train_images.append(all_images[index])

test_images = []
for index in test_index:
    test_images.append(all_images[index])

#------------------------------------------------------------------
def create_codebook(images):
    orb = cv2.ORB_create()
    features  = []
    for image in images:
        im =  cv2.imread('./data/lsp_dataset/images/'+image)
        im = np.rot90(im,2)
        kp, des = orb.detectAndCompute(im,None)
        if des is not None:
            features.extend(des)
    kmeans = cluster.KMeans(n_clusters = codebook_words, n_init=10)
    kmeans.fit(features)
    codebook = kmeans.cluster_centers_
    print("Codebook generated...")
    np.save('codebook_augmented.npy',codebook)

# create_codebook(train_images)
#------------------------------------------------------------------

codebook = np.load('codebook_combined.npy',allow_pickle=True)
print("Codebook Loaded...")

#------------------------------------------------------------------
def divide_regions(image):
    height, width = image.shape[:2]
    #dividing into 4 sub-regions
    #1
    start_row, start_col = int(0), int(0)
    end_row, end_col = int(height *.5), int(width*.5)
    region_1 = image[start_row:end_row , start_col:end_col]
    #2
    start_row, start_col = int(0), int(width * .5)
    end_row, end_col = int(height*.5), int(width)
    region_2 = image[start_row:end_row , start_col:end_col]
    #3
    start_row, start_col = int(height*.5), int(0)
    end_row, end_col = int(height), int(width*.5)
    region_3 = image[start_row:end_row , start_col:end_col]
    #4
    start_row, start_col = int(height *.5), int(width*.5)
    end_row, end_col = int(height), int(width)
    region_4 = image[start_row:end_row , start_col:end_col]

    return region_1,region_2,region_3,region_4
#------------------------------------------------------------------

#------------------------------------------------------------------
def get_region_feature(des):
    #knn
    hist = np.zeros(codebook_words)
    for d in des:
        count = 0
        min_count = 0
        histogram = []    
        min_dist = sys.maxsize
        for word in codebook:
            dist = np.linalg.norm(word-np.array(d),2)
            if dist<min_dist:
                min_count = count
                min_dist = dist
            count += 1
        hist[min_count] += 1
    return hist


def extract_input_features(images):
    orb = cv2.ORB_create()  
    INPUT_TO_SVM = []
    #extracting features from each sub-region
    for image in images:
        im =  cv2.imread('./data/lsp_dataset/images/'+image)
        im = np.rot90(im,2)
        regions = divide_regions(im)
        this_image_feature  = []
        for region in regions:
            kp, des = orb.detectAndCompute(region,None)
            if des is not None:
                subfeature = get_region_feature(des)
                this_image_feature.extend(subfeature)
            else:
                this_image_feature.extend(np.zeros((codebook_words,)))  
        INPUT_TO_SVM.extend([this_image_feature])
    return INPUT_TO_SVM
    
#-----------------------------------------------------------------------------------

training_input_augmented = np.array(extract_input_features(train_images))
np.save('training_input_augmented.npy',training_input_augmented)

print("Input Generated.")
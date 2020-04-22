import sklearn.cluster as cluster
import detect_upper_body as ubody
import cv2
import sys

def create_codebook(path):
    orb = cv2.ORB_create()
    dataset_path = ""
    images = []
    features  = []
    for image in images:
        boxes = ubody.get_upper_bodies(image)
        for box in boxes:
            kp, des = orb.detectAndCompute(box,None)
            if des is not None:
                features.extend(des)
    kmeans = cluster.KMeans(n_clusters = 400, n_init=10)
    kmeans.fit(features)
    codebook = kmeans.cluster_centers_
    #save codebook to disk
   
    

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



def get_region_feature(des,codebook):
    #knn
    hist = np.zeros(400)
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


def extract_inpput_features(path):
    orb = cv2.ORB_create()
    dataset_path = ""
    images = []
    codebook = None #load from disk    
    INPUT_TO_SVM = []
    #extracting features from each sub-region
    for image in images:
        boxes = ubody.get_upper_bodies(image)
        this_image_features = []
        for box in boxes:
            regions = divide_regions(box)
            this_box_feature  = []
            for region in regions:
                kp, des = orb.detectAndCompute(region,None)
                if des is not None:
                    subfeature = get_region_feature(des,codebook)
                    this_box_feature.extend(subfeature)
            concatenated_bounding_box_feature = np.hstack([this_box_feature[0],this_box_feature[1],this_box_feature[2],this_box_feature[3]])
            this_image_features.extend(this_box_feature)
    INPUT_TO_SVM.extend(this_image_features)
    return INPUT_TO_SVM
    
        
            

    


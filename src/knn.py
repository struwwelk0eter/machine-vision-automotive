import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def train_vocabulary():
    #looks at all images and calculates suitable clusters based on the keypoints
    k_means_trainer = cv2.BOWKMeansTrainer(100)
    for filename in os.listdir(r'C:\train'):
        path_to_image = "{}{}".format(r'C:\train', filename)
        img = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
        surf = cv2.xfeatures2d.SURF_create()
        print (path_to_image)
        #plt.imshow(img, cmap='gray')
        key_points, descriptors = surf.detectAndCompute(img, None)

        if len(key_points) <= 0:
            continue
        descriptors = np.float32(descriptors)

        k_means_trainer.add(descriptors)
    global vocabulary
    vocabulary = k_means_trainer.cluster()
    print ('Vocabulary trained successfully!')
    #return vocabulary
    #print (vocabulary)
    
def get_visual_word_histogram_of_images(vocabulary):
    #takes all pictures and assigns the found keypoints of each picture to the clusters from the vocabulary --> each picture is described --> feature
    surf = cv2.xfeatures2d.SURF_create()
    bow_ext = cv2.BOWImgDescriptorExtractor(surf, cv2.BFMatcher(cv2.NORM_L2))
    bow_ext.setVocabulary(vocabulary)
    
    global bow_list
    bow_list = []
    i = 0
    for filename in os.listdir(r'C:\train'):
        path_to_image = "{}{}".format(r'C:\train', filename)
        image = cv2.imread(path_to_image, 0)
        kp, des = surf.detectAndCompute(image, None)
        global histogram
        histogram = bow_ext.compute(image, kp)[0]
        bow_list.append(histogram)
        i = i+1
        print (i)
    
    #return bow_list
    print (bow_list)
    print (histogram.shape)
    xachse = list(range(0, 100, 1))
    plt.plot(xachse, histogram)
    plt.show()

def knn_opencv(): #opencv
    trainData = np.array(bow_list)
    trainData.astype(int)
    #print (trainData)
    responses = np.array([[0],[1],[0],[0],[0],[0],[1],[1],[1],[1]])
    #0 =good, 1 = bad
    responses.astype(int)
    #print (responses)
    
#    red = trainData[responses.ravel()==0]
#    plt.scatter(red[:,0],red[:,1],80,'r','^')
#    blue = trainData[responses.ravel()==1]
#    plt.scatter(blue[:,0],blue[:,1],80,'b','s')
#    print (red)
#    plt.show()

    #NEWCOMER: 1=gutklammer, 2=mittel, 3=mittelklammer, 4=schlechtklammer, 5=schlechtschraeg - testimages
    newcomer1 = np.array([newcomer1list]).astype(np.float32)
    newcomer2 = np.array([newcomer2list]).astype(np.float32)
    newcomer3 = np.array([newcomer3list]).astype(np.float32)
    newcomer4 = np.array([newcomer4list]).astype(np.float32)
    newcomer5 = np.array([newcomer5list]).astype(np.float32)
    #newcomer = np.random.randint(0,100,(1,2))
    #newcomer.astype(int)
    print (trainData.shape)
    print (responses.shape)
    #print (newcomer.shape)
    knn = cv2.ml.KNearest_create()
    knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)
    ret, results, neighbours ,dist = knn.findNearest(newcomer4, 3)

    print ("result: ", results,"\n")
    print ("neighbours: ", neighbours,"\n")
    print ("distance: ", dist)

def knn_sklearn(): #sklearn
    trainData = np.array(bow_list)
    trainData.astype(int)
    #print (trainData)
    responses = np.array([0,1,0,0,0,0,1,1,1,1])
    #0 =good, 1 = bad
    responses.astype(int)
    
    #NEWCOMER: 1=gutklammer, 2=mittel, 3=mittelklammer, 4=schlechtklammer, 5=schlechtschraeg - testimages
    newcomer1 = np.array([newcomer1list]).astype(np.float32)
    newcomer2 = np.array([newcomer2list]).astype(np.float32)
    newcomer3 = np.array([newcomer3list]).astype(np.float32)
    newcomer4 = np.array([newcomer4list]).astype(np.float32)
    newcomer5 = np.array([newcomer5list]).astype(np.float32)

    print (trainData.shape)
    print (responses.shape)
    #print (newcomer.shape)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(trainData, responses)
    result = knn.predict(newcomer5)

    print ("result:", result,"\n")
    
def testingdata():
    surf = cv2.xfeatures2d.SURF_create()
    bow_ext = cv2.BOWImgDescriptorExtractor(surf, cv2.BFMatcher(cv2.NORM_L2))
    bow_ext.setVocabulary(vocabulary)
    
#    global bow_list
#    bow_list = []
    i = 0
    global train_list
    train_list = []
    for filename in os.listdir(r'C:\test'): 
        path_to_image = "{}{}".format(r'C:\test', filename)
        image = cv2.imread(path_to_image, 0)
        kp, des = surf.detectAndCompute(image, None)
        global histogramtrain
        histogramtrain = bow_ext.compute(image, kp)[0]
#       bow_list.append(histogram)
        train_list.append(histogramtrain)
        i = i+1
        print (i)
        print(path_to_image)
    
    global newcomer1list, newcomer2list, newcomer3list, newcomer4list, newcomer5list
    newcomer1list = train_list[0]
    newcomer2list = train_list[1]
    newcomer3list = train_list[2]
    newcomer4list = train_list[3]
    newcomer5list = train_list[4]
    
    print (train_list[0])
    print (train_list[1])
    print (train_list)
    xachse = list(range(0, 100, 1))
    plt.plot(xachse, histogramtrain)
    plt.show()
    
#1
train_vocabulary()

#2
get_visual_word_histogram_of_images(vocabulary)

#3
testingdata()

#4
knn_sklearn()

#a = np.array(histogramtrain)
#print (a)

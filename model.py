import tensorflow as tf
import numpy as np
import cv2

def create_label(image_name):
    name = image_name.split('.')[0]
    if name=="Rabindra":
        return np.array([1,0,0])
    elif name=="Riju":
        return np.array([0,1,0])
    elif name=="Rojina":
        return np.array([0,0,1])


##Create Data 
import os
from random import shuffle
from tqdm import tqdm



def mydata():
    data=[]
    for image in tqdm(os.listdir("dataset")):
        path=os.path.join("dataset",image)
        image_data =cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        image_data =cv2.resize(image_data, (50,50))
        data.append([np.array(image_data),create_label(image)])

    shuffle(data)
    return data 


data = mydata()



train = data[:2400]
test = data[2400:]
x_train = np.array([i[0]  for i in train]).reshape(-1,50,50,1)
y_train = np.array([i[1] for i in train])
print(x_train.shape)
x_test = np.array([i[0] for i in test]).reshape(-1,50,50,1)
y_test = np.array([i[1] for i in test])
print(x_train.shape)
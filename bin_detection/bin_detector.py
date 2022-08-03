import cv2 as cv
import pandas as pd
import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt; plt.ion()
from skimage.segmentation import mark_boundaries
import os, cv2
from skimage.measure import label, regionprops

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(w, X, y):
    h = sigmoid(X @ w)
    m = len(y)
    cost = 1 / m * np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h))
    grad = 1 / m * ((y - h) @ X)
    return cost, grad

def training(X,y,max_iter=5000,Lr=0.001):
    all_costs = []
    classes = np.unique(y)
    costs = np.zeros(max_iter)
    w = np.zeros(X.shape[1])
    for i in range(max_iter):
        costs[i], grad = cost(w, X, y)
        w += Lr * grad
        all_costs.append(costs)
    
    print(w)
    return w, all_costs

def predict(w,X):
    m = X.shape[0]
    y = np.zeros(m)
    
    A = sigmoid(X @ w)
    print(A.shape)
    
    for i in range(A.shape[0]):
        if A[i] >= 0.5:
            y[i] = 1
    return y

thetas = [-0.51719884, -0.25125233,  0.4795692]
w = np.array(thetas)

def segment_image(img):
    X_ = cv2.imread(img)
    X_ = cv2.cvtColor(X_, cv2.COLOR_BGR2RGB)
    a = X_.shape[0]
    b = X_.shape[1]
    c = a*b
    X=X_.reshape(c,3)
    
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    m = X.shape[0]
    y = np.zeros(m)

    A = sigmoid(X @ w)

    for i in range(A.shape[0]):
        if A[i] >= 0.5:
            y[i] = 1

    mask_y = y.reshape(a,b)
    mask_y = np.multiply(mask_y,255)
    
    kernel = np.ones((5,5), np.uint8)
    mask_y = cv2.erode(mask_y, kernel, iterations=3)
    mask_y = cv2.dilate(mask_y, kernel, iterations=3)
    mask_image = cv2.imshow('',mask_y)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #bounding box part
    mask_labels = label(mask_y)
    props = regionprops(np.asarray(mask_labels))
    
    img_copy = np.asarray(mask_y)
    for prop in props:
        if prop.bbox_area > 1:
            cv2.rectangle(X_, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (145, 0, 0), 2)
            cv2.rectangle(img_copy, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (145, 0, 0), 2)
            

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 5))
    ax1.imshow(X_)
    ax1.set_title('Image')
    ax2.set_title('Mask')
    ax3.set_title('Image with derived bounding box')
    ax2.imshow(X_, cmap='gray')
    ax3.imshow(mask_y)
    plt.show()
    
    return mask_image

def get_bounding_boxes(self, img):
    
    mask = img
    x_max, y_max = mask.shape[0], mask.shape[1]
    contours, hierarchy = cv2.findContours(mask[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Find Contours
    
    boxes = []
    for cnt_num in contours:
	    x, y, w, h = cv2.boundingRect(cnt_num)
        area_ratio = cv2.contourArea(cnt_num) / (y_max * x_max)
	    size_ratio = h/w
	    if 0.75 <= size_ratio <= 3.5 and area_ratio > 0.005:
				boxes.append([x, y, x + w, y + h])

    return boxes
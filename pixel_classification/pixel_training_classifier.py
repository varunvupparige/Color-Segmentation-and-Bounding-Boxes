import pandas as pd
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import os, cv2
from numpy import savetxt
from sklearn.metrics import classification_report


# Generates RGB data from images

def read_pixels(folder, verbose = False):
    n = len(next(os.walk(folder))[2]) # number of files
    X = np.empty([n, 3])
    i = 0
  
    if verbose:
        fig, ax = plt.subplots()
        h = ax.imshow(np.random.randint(255, size=(28,28,3)).astype('uint8'))
    
    for filename in os.listdir(folder):
        if ".png" in filename:
            img = cv2.imread(os.path.join(folder,filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X[i] = img[0,0].astype(np.float64)/255
            i += 1
    
    # display
    if verbose:
        h.set_data(img)
        ax.set_title(filename)
        fig.canvas.flush_events()
        plt.show()

    return X

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(w, X, y):
    h = sigmoid(X @ w)
    m = len(y)
    cost = 1 / m * np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h))
    grad = 1 / m * ((y - h) @ X)
    return cost, grad

def training(X,y,max_iter=5000,Lr=0.001):
    ws = []
    all_costs = []
    classes = np.unique(y)
    
    for c in classes:
        costs = np.zeros(max_iter)
        #one vs rest classifier
        bin_y = np.where(y==c,1,0)
        
        w = np.zeros(X.shape[1])
        for i in range(max_iter):
            costs[i], grad = cost(w, X, bin_y)
            w += Lr * grad
            
        ws.append(w)
        all_costs.append(costs)

    print(ws)
    return ws, all_costs 


folder = 'data/training'
X1 = read_pixels(folder+'/red')
X2 = read_pixels(folder+'/green')
X3 = read_pixels(folder+'/blue')
y1, y2, y3 = np.full(X1.shape[0],1), np.full(X2.shape[0], 2), np.full(X3.shape[0],3)
X, y = np.concatenate((X1,X2,X3)), np.concatenate((y1,y2,y3))
print(X.shape)
savetxt('part1X.csv',X,delimiter=',')
iter_ = 20000
ws, all_costs = training(X,y,max_iter=iter_,Lr=0.005)

def classify(ws,X):
    m = X.shape[0]
    y = np.zeros(m)
    ws = ws.reshape(3, 3)
    print(ws)
    
    A = sigmoid(X @ ws)
    
    for i in range(A.shape[0]):
        y[i] = np.argmax(A[i, :]) + 1
    return y


plt.plot([i for i in range(iter_)], all_costs[0])
plt.show()

y_pred = classify(np.array(ws), X)
np.count_nonzero(y==1),np.count_nonzero(y==2),np.count_nonzero(y==3)
print(classification_report(y, y_pred, labels=[1,2,3], target_names=["Red", "Green", "Blue"]))

'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''


import os, cv2
from roipoly import RoiPoly
import numpy as np
from numpy import asarray
from numpy import savetxt
from matplotlib import pyplot as plt

if __name__ == '__main__':
    folder = 'data/training'
    filename = '0057.jpg'  
    img = cv2.imread(os.path.join(folder,filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  
  # display the image and use roipoly for labeling
    fig, ax = plt.subplots()
    ax.imshow(img)
    my_roi = RoiPoly(fig=fig, ax=ax, color='r')
  
  # get the image mask
    mask = my_roi.get_mask(img)
    npmask = np.array(mask)
    X=img[mask==1]
    savetxt('mask21x.csv', X, delimiter=',')
  
  # display the labeled region and the image mask
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('%d pixels selected\n' % img[mask,:].shape[0])
  
    ax1.imshow(img)
    ax1.add_line(plt.Line2D(my_roi.x + [my_roi.x[0]], my_roi.y + [my_roi.y[0]], color=my_roi.color))
    ax2.imshow(mask)
    plt.show(block=True)


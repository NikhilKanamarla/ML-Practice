import sys
import numpy as np
from PIL._util import *
from scipy.misc import imread, imsave, imresize
import matplotlib.pyplot as plt

if __name__ == '__main__':

    x = np.array([1,2,3])
    #number of dimensions (r,c)
    print(x.shape)
    #print a specific index
    print(x[1])
    x2 = np.array([[1,2,3],[4,5,6]])
    print(x2.shape)
    #index access
    print(x2[1][0])
    #create a pre filled array
    a = np.zeros((3,2))
    print(a)

    #slicing allows you to access a part of an array
    a2 = np.array([[1,2,3],[4,5,6],[7,8,9]])
    #[startRow:endRow,startCol:endCol] inclusive start and exclusive end 
    array = a2[0:2,0:2]
    print(array)
    #modifying a slice changes the orginal array
    array [0] [0] = 1000
    print(array)
    #access a specific row (: means all)
    array2 = a2[2,:]
    print(array2)
    #slice based on a condition
    condition = (a2 > 5)
    greaterA = a2[condition]
    print(greaterA)
    z2 = np.array([[1,2,3,4,5,6],[1,2,3,4,5,6]])
    temp = z2[:3]
    print("check here ", temp)

    #pick specific columns in an array
    print(a2)
    #columns to be chosen
    tester = np.array([0,1,0])
    store = a2[np.arange(3),tester]
    print(store)

    #math
    x = np.array([[1,2,3],[4,5,6],[7,8,9]])
    y = np.array([[10,11,12],[13,14,15],[16,18,18]])
    z = x + y
    print(z)
    z1 = x - y
    print(z1)
    z2 = np.sqrt(x)
    print(z2)
    z3 = x / y
    print(z3)
    #sum all elements in array
    print(np.sum(z2))
    #matrix multiplication
    print(np.dot(z2,z1))
    #transpose 
    z4 = x.T
    print(z4)

    #Broadcasting allows you do manipulate arrays of different sizes 
    d = np.array([[1,2,3],[4,5,6]])
    e = np.array([1,2,3])
    #create an array with the same shape as d
    f = np.empty_like(d)
    # fill array 
    for i in range(len(f)):
        f[i,:] = d[i,:] + e 
    print(f)
    #other method of brodcasting
    f2 = d + e
    #broadcasting arrays of different sizes then the larger array is kept (one dimension must match)
    print(f2)
    print(d.shape)
    #change the shape of a numpy array
    #np.reshape(array,(row,columns))
    reshaped = np.reshape(d,(6,1))
    print(reshaped)


    #image operations(scipy)
    img = imread("/home/nikhil/Downloads/ForwardObserverDocuments/cowc-m/ProcessedData/retinanet_results/6.png")
    #image length/width/channels(pixels)
    print(img.shape)
    #change the tint
    img = img * [0.9,0.9,0.9]
    #resize the image 
    img = imresize(img,(200,200))
    imsave("/home/nikhil/Downloads/ForwardObserverDocuments/cowc-m/ProcessedData/retinanet_results/6B.png",img)

    #matplot 
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    x = np.arange(0, 3 * np.pi,0.1)
    x = np.sin(x)
    plt.plot(x)
    plt.show()
    plt.savefig("/home/nikhil/Downloads/matplot.png")
    #using matplot to display images 
    plt.imshow(img)
    plt.savefig("/home/nikhil/Downloads/matplot2.png")

    store = np.empty([992,1])
    store [0] = 5
    print(store.shape)
    print(store)

    xd = np.array([0,1,2,3,4])
    print(xd.size)


    boxes = np.array([[111,242,132,143],[151,171,165,200]])
    boxes = boxes * 0.15
    print(boxes)
    num = 37.53
    num = round(num,1)
    print(num)

    num = '000000'
    frame = 10
    length = len(str(frame))
    num = num[:len(num)-length]
    path = num + (str(frame))
    print(path)

    

    
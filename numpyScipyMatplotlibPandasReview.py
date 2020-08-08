import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import tensorflow as tf
import keras
from sklearn.metrics import roc_auc_score
from scipy.misc import imread, imsave, imresize


#Review dictionaries, arrays, loops, np arrays, pandas 
if __name__ == '__main__':
    #numpy array 
    x = np.array([1,2,3,4,5,6])
    #get rows and columns 
    shape = x.shape
    print(shape)
    x[1] = 5
    x2 = np.array([[1,2,3,4,5,6],[12,1,3,14,1,5]])
    rows,columns = x2.shape #(rows,columns)
    print(rows, columns)
    #size
    size = np.size(x)
    #prefilled array with zeros
    emptyZero = np.zeros((rows,columns))
    print(emptyZero)
    #slices
    slice1 = x2[x2 > 5]
    print(slice1)
    #[startRow:endRow,startCol:endCol] inclusive start and exclusive end 
    slice2 = x2[0:5,2:4]
    print(slice2) 
    #reshape a numpty array 
    x2 = np.reshape(x2,(12,1))
    print(x2[11][0])
    #iterate through
    for item in x2:
        print(item)
    #start, stop, step     
    slice = np.arange(x[0],x[-1],3)
    print(slice)
    x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
    v = np.array([1, 0, 1])
    #creates numpy array with same shape as x
    y = np.empty_like(x)
    #broadcasting a numpy array
    list = [0,1,2,3]
    for z in list:
        y[z,:] = x [z,:] + v[:]
    print(y) 
    basic = np.zeros((1,1,4))
    print(basic)
    list = [[243,212,120,175],[243,255,322,144]]
    temp = np.array(list)
    print(temp)


    #scipy
    img = imread('/home/nikhil/Downloads/ForwardObserverDocuments/cowc-m/ProcessedData/Toronto_ISPRS/test/Toronto_03559.22.19.png')
    print(img.shape)
    tinted_img = img * [0.5,0.5,0.9]
    tinted_img = imresize(tinted_img,[256,256])
    imsave("testScipy.png",tinted_img)

    #matplotlib 
    x = np.arange(0, 3 * np.pi, 0.1)
    y_sin = np.sin(x)
    y_cos = np.cos(x)

    # Plot the points using matplotlib
    #plot is the line
    plt.plot(x, y_sin)
    plt.plot(x, y_cos)
    #label for x and y axis
    plt.xlabel('x axis label')
    plt.ylabel('y axis label')
    # title and legend 
    plt.title('Sine and Cosine')
    plt.legend(['Sine', 'Cosine'])
    plt.savefig('matplotlibTest')

    #pandas tutorial
    #store data in CSV 
    dataFrame = pd.read_csv('pokemon_data.csv')
    #get data 
    print(dataFrame.head(5))
    print(dataFrame.tail(5))
    #get columns
    columns = len(dataFrame.columns)
    print('num columns ', columns)
    specifcColumn = dataFrame['Type 1']
    print(specifcColumn)
    #store multiple columns 
    manyCols = dataFrame[['Type 1','Attack']]
    print("many columns ", manyCols)
    #get first row or any row specify 
    rowsFirst = dataFrame.iloc[[0]]
    print('the first row ', rowsFirst)
    #get specific rows 
    storeNarrow = dataFrame.loc[dataFrame['Type 1'].str.contains('Grass')]
    print(storeNarrow)
    #remove columns
    dataFrame = dataFrame.drop(columns=['Generation'])
    #export to CSV
    dataFrame.to_csv("pokemon_data2.csv",index=False,sep=",")
    #copy
    dataFrame2 = dataFrame.copy()

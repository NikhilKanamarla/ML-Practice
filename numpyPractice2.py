import sys
import numpy as np
from PIL._util import *
from scipy.misc import imread, imsave, imresize
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image, ImageDraw


if __name__ == '__main__':
    a = np.array([[1,2,3],[6,4,5]])
    #start (inclusive), stop (exclusive), step size
    b = np.arange(start=0,stop=10,step=2)
    print(b)
    #all rows and certain columns 
    slice = a[:, np.arange(start=0,stop=3,step=2)]
    print(slice)
    #change the shape
    slice = slice.reshape((4,1))
    print(slice)
    x = np.array([1,2,3,4,5,6])
    #condition, return if true, return if false
    slice2 = np.where(x > 2,x,0)
    print(slice2)

    #PIL
    im = Image.open("/home/nikhil/Downloads/ForwardObserverDocuments/cowc-m/ProcessedData/retinanet_results/6.png")
    print(im.format,im.size,im.mode)
    im.show()
    box = (10,10,10,10)
    region = im.crop(box)
    region = region.transpose(Image.ROTATE_180)
    im.paste(region,box)
    im.resize((100,100))
    im.save("test.png")
    im2 = ImageDraw.Draw(im)

    #Numpy Scipy multi dimensional image processing
    f = misc.face()
    misc.imsave('face.png', f) # uses the Image module (PIL)
    plt.imshow(f)
    plt.savefig("tester.png")



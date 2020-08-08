import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt 

if __name__ == '__main__':
    print("Hello world")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
    # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    print(tf.version)
    #data set from keras api
    data = keras.datasets.fashion_mnist
    #data being seperated 
    (train_images,train_labels) , (test_images, test_labels) = data.load_data()
    #labels
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    print(train_labels[6])
    #get pixel values between 0 and 1
    train_images = train_images/255
    test_images = test_images/255

    #flattening the data means combining the data sent into input layer 
    #input layer is 784 nodes for each pixel in the 28 * 28 image
    #hidden layer is 128 nodes with weights and biases between input and output nodes 
    #output layer is 10 nodes for the number of labels with an activation score for each node

    #defining the model in terms of input,hidden and output layer and activation
    #keras.sequential just mean a sequence of layers 
    #keras.flatten specifes that input must be simplified to a 1D array, input_shape is the size of the image
    #keras.layers.Dense is a fully connected layer (weights), activation function is recitfy liner unit 
    #activation softmax picks values for each node such that all values add up to one (probability)
    model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)), keras.layers.Dense(128,activation="relu"), 
    keras.layers.Dense(10,activation = "softmax")])

    #specify parameters for model training
    #loss calcuates error of function using metrics
    #optimizer changes the weights in the neural network according to loss calculation  
    model.compile(optimizer="adam",loss = "sparse_categorical_crossentropy",metrics=["accuracy"])

    #train model
    #epochs is the number of times we feed info to neural network 
    #pass in training images and labels 
    model.fit(train_images,train_labels,epochs=5)

    #testing model using test labels and images
    (test_loss, test_acc) = model.evaluate(test_images,test_labels)
    print("Tested accuracy ", test_acc)

    #making prediction using model
    prediction = model.predict(test_images)
    #picks neuron with highest score according to softmax probability for the first image
    print(class_names[np.argmax(prediction[0])])

    #making prediction and comparing to actual label
    #plt grid is visual depicition
    for x in range(5):
        plt.grid(False)
        plt.imshow(test_images[x],cmap=plt.cm.binary)
        plt.xlabel("Actual " + class_names[test_labels[x]])
        plt.title("Prediction " + class_names[np.argmax(prediction[x])])
        plt.show()


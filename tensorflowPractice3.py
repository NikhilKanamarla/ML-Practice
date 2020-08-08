import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt 

#returns keys for a piece of text
def decode_review(text):
    return " ".join([reverse_word_index.get(i,"?") for i in text])

#formatting text
def review_encode(s):
    encoded = [1]
    for word in s:
        if word in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded

if __name__ == '__main__':
    print("hello world")
    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

    #load data
    data = keras.datasets.imdb

    #seperate data and only take 10,000 most frequent words 
    (train_data, train_labels), (test_data,test_labels) = data.load_data(num_words=10000)

    #gives dictionary of each word and the number of times it occurs
    word_index = data.get_word_index()
    #splitting into keys and values
    #adds 3 to the count of every word
    word_index = {k:(v+3) for k, v in word_index.items()}
    #helps format the data 
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3
    #normalize data to 250 words per post (maxLen)
    #padding end of shorter posts by adding space at end (padding = post)
    train_data = keras.preprocessing.sequence.pad_sequences(train_data,value= word_index["<PAD>"], padding="post",maxlen=250)
    test_data = keras.preprocessing.sequence.pad_sequences(test_data,value= word_index["<PAD>"], padding="post",maxlen=250)
    #swaps the values for the keys so intger points to word
    reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])
    print(decode_review(test_data[0]))

    #define model structure
    model = keras.Sequential()
    #Embedding layer tries to group similar words
    #generates 10000 word vectors and uses as data to pass to next layer and groups word vectors based on similarity  
    model.add(keras.layers.Embedding(880000,16))
    #scale down to one dimension (GlobalAveragePooling)
    model.add(keras.layers.GlobalAveragePooling1D())
    #first hidden layer and dense means fully connected layer and tries to classify words 
    model.add(keras.layers.Dense(16,activation="relu"))
    #output layer with one neuron with confidence level from one to zero measuring how postive or negative review is 
    model.add(keras.layers.Dense(1,activation="sigmoid"))

    #training parameters
    model.compile(optimizer="adam",loss ="binary_crossentropy",metrics=["accuracy"])

    #validation data checks how model is performing during training
    #first 10,000 entries is for validation and next entries are for training 
    x_val = train_data[:10000]
    x_train = train_data[10000:]
    y_val = train_labels[:10000]
    y_train = train_labels[10000:]

    #train model
    #batch size is how many posts send into neural network at once
    #validation data tests how model is being trained 
    fitModel = model.fit(x_train,y_train,epochs=40,batch_size=512,validation_data=(x_val,y_val),verbose=1)

    #testing model performance
    result = model.evaluate(test_data,test_labels)
    print(result)

    #predicting a specific post negative or postive
    #get a specific post 
    test_review = test_data[0]
    #make prediction
    predict = model.predict([test_review])
    print("Review ")
    print(decode_review(test_review))
    print("Prediction " + str(predict[0]))
    print("Actual " + str(test_labels[0]))

    #saving and loading models
    model.save("model.h5")
    #all previous code code be commented out and then use and load the model
    model = keras.models.load_model("model.h5")
    #prediction using outside data
    with open("socialNetworkReview.txt") as f:
        for line in f.readlines():
            #preprocess data 
            nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")","").replace(":","").replace("\"","").strip().split(" ")
            encode = review_encode(nline)
            encode = keras.preprocessing.sequence.pad_sequences([encode],value= word_index["<PAD>"], padding="post",maxlen=250)
            model.predict(encode)
            print(line)
            print(encode)
            print(predict[0])






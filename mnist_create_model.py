# example of loading the mnist dataset
from keras.datasets import mnist
# from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


# load dataset
(trainX, trainy), (testX, testy) = mnist.load_data()

# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))

# plot first few images
# for i in range(9):
#  	# define subplot
#  	pyplot.subplot(330 + 1 + i)
#  	# plot raw pixel data
#  	pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))
# # show the figure
# pyplot.show()

# printing labels
# for i in range(9):
#     print(trainy[i])

# reshape dataset to have a single color channel,
# 4-dim to deal with keras API, (trainX.shape[0] = 60000, 28, 28, 1) 
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))
# normalize the image data by dividing each pixel value by 255 
# (since RGB value can range from 0 to 255):
trainX = trainX / 255
testX = testX / 255

# one hot encode target values 
# transforming the integer into a 10 element binary vector
trainy = to_categorical(trainy)
testy = to_categorical(testy)

# define CNN model
"""
Convolutional Layers: 
The convolutional layer is the very first layer 
where we extract features from the images in our datasets
_____
Pooling Layer: 
When constructing CNNs, it is common to insert pooling layers 
after each convolution layer to reduce 
the spatial size of the representation to reduce the parameter counts 
which reduces the computational complexity. 
In addition, pooling layers also helps with the overfitting problem. 
Basically we select a pooling size to reduce the amount of the parameters 
by selecting the maximum, average, or sum values inside these pixels
_____
A Set of Fully Connected Layers
A fully connected network is our RegularNet where each parameter 
is linked to one another to determine the true relation 
and effect of each parameter on the labels. 
Since our time-space complexity is vastly reduced thanks to convolution 
and pooling layers, we can construct a fully connected network in the end 
to classify our images
_____
Dropout layers fight with the overfitting by disregarding 
some of the neurons while training
_____ 
while Flatten layers flatten 2D arrays to 1D arrays 
before building the fully connected (Dense)layers.

"""

model = Sequential() # It creates an empty model object.

model.add(Conv2D(32, kernel_size=(3, 3),
      activation='relu',
      input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu')) # to interpret the features
"""
ReLU :
activation function that looks and acts like a linear function, 
but is, in fact, a nonlinear function allowing 
complex relationships in the data to be learned.
___
ReLU(x):
    if x > 0:
        return x
    else:
        return 0.0
if x(input) positive and not equal zero -> x
if x(input) negative or equal zero -> 0.0

"""

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax')) # output layer , 10 -> numbers(0-9)
# The “softmax” activation is used when we’d like 
# to classify the data into a number of pre-decided classes.

model.compile(loss='categorical_crossentropy',
      optimizer='adam',
      metrics=['accuracy'])
"""
loss function: 
Use sparse categorical crossentropy: when your classes are mutually exclusive 
(e.g. when each sample belongs exactly to one class) 
Use categorical crossentropy: when one sample can have multiple classes or labels 
are soft probabilities (like [0.5, 0.3, 0.2]). multi-class classification
The difference between sparse_categorical_crossentropy and 
categorical_crossentropy is whether your targets are one-hot encoded.
____
optimizer function:
Adam Optimizer 
Adaptive Moment Estimation is an algorithm for optimization technique 
for gradient descent. 
The method is really efficient when working with large problem 
involving a lot of data or parameters. 
It requires less memory and is efficient. Intuitively, 
it is a combination of the ‘gradient descent with momentum’ algorithm 
and the ‘RMSP’ algorithm.
"""
model.fit(trainX, trainy,
          epochs=10,
          batch_size=32,
          verbose=0,
          validation_data=(testX, testy))

# epoch: is one forward pass and one backward pass of all training examples
# batch size is the number of training examples in one forward or backward pass

score = model.evaluate(testX, testy, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save("mnist_test_model.h5")









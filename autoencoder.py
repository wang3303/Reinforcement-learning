import numpy as np 
np.random.seed(1)

from keras.datasets import mnist
from keras.models import Model 
from keras.layers import Dense, Input
import matplotlib.pyplot as plt

(x_train, _), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255. - 0.5       # minmax_normalized
x_test = x_test.astype('float32') / 255. - 0.5         # minmax_normalized
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))
print(x_train.shape)
print(x_test.shape)

dim = 2

input_img = Input(shape=(784,))
encoded = Dense(128,activation = 'relu')(input_img)
encoded = Dense(64, activation = 'relu')(encoded)
encoded = Dense(10, activation = 'relu')(encoded)
encoder_output = Dense(dim,)(encoded)

decoded = Dense(10,activation = 'relu')(encoder_output)
decoded = Dense(64,activation = 'relu')(decoded)
decoded = Dense(128,activation = 'relu')(decoded)
decoded = Dense(784,activation = 'tanh')(decoded)

autoencoder = Model(input = input_img,output = decoded)
encoder = Model(input = input_img, output = encoder_output)

autoencoder.compile(optimizer = 'adam', loss = 'mse')

autoencoder.fit(x_train,x_train,nb_epoch = 15,batch_size = 256, shuffle = True)

encoded_imgs = encoder.predict(x_test)
plt.scatter(encoded_imgs[:,0],encoded_imgs[:,1],c = y_test)
plt.show()

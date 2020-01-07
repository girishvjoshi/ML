# Author: Girish Joshi
# Date: 01/06/2020
# This is the code for training and visulizing a Variation Autoencoder

import numpy as np 
import tensorflow as tf 
from VAE_net import VAE
import matplotlib.pyplot as plt 

from tensorflow.examples.tutorials.mnist import input_data
database = input_data.read_data_sets('./data', one_hot = True)

learning_rate = 0.001 #learning rate
epochs = 30000 # Training epochs
batch_size = 32 # Training batch size
hidden_layer_size = 512 #Hidden layer size
latent_layer_size = 2 # Laten layer size

n = 20
x_limit = np.linspace(-2,2,n)
y_limit = np.linspace(-2,2,n)

empty_image = np.empty((28*n, 28*n))

def main():

    with tf.Session() as sess:
        agent = VAE(sess, learning_rate, hidden_layer_size, latent_layer_size)
        sess.run(tf.global_variables_initializer())

        for steps in range(epochs):
            x_batch,_ = database.train.next_batch(batch_size)
            loss = agent.train(x_batch)
            if steps % 1000 == 0 and steps>0:
                print('Epoch:',steps, 'Training Loss:', loss)

        # Testing The VAE network with noise inputs
        for i, zi in enumerate(x_limit):
            for j, pi in enumerate(y_limit):
                generate_latent_layer = np.array([[zi,pi]])
                generated_images = agent.decode(generate_latent_layer)
                empty_image[(n-i-1)*28:(n-i)*28, j*28:(j+1)*28] = generated_images[0].reshape(28,28)

    plt.figure(2)
    plt.figure(figsize=(8,10))
    x,y = np.meshgrid(x_limit, y_limit)
    plt.imshow(empty_image, origin='upper', cmap='gray')
    plt.grid(False)
    plt.show()


if __name__ == '__main__':
    main()

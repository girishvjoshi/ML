# Author: Girish Joshi
# Date: 01/06/2020
# This is the code for Initialize a Variation Autoencoder netowrk

import numpy as np 
import tensorflow as tf

# Using Vavier Initialzation for network weights biases
def weight_variable(shape):
    initial_val = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initial_val(shape))

# Defining VAE class file
class VAE(object):

    def __init__(self, sess, lr, hidden_layer_size, latent_layer_size):
        self.sess = sess
        self.lr = lr # Learning rate
        self.hLayer_N = hidden_layer_size #Hidden Layer size
        self.Latent_layern_N = latent_layer_size  # Latent Layer size
        self._placeholder()
        self._init_net()
        self._loss()
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)#tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
        

    def _placeholder(self):
        self.inputs = tf.placeholder(tf.float32, (None, 784), name='Inputs')
        self.noise_inputs = tf.placeholder(tf.float32, (None, self.Latent_layern_N), name='Noise')

    def _init_net(self):
        self.encoder_w1 = weight_variable([784, self.hLayer_N])
        self.encoder_b1 = weight_variable([self.hLayer_N])
        self.encoder_w2m = weight_variable([self.hLayer_N, self.Latent_layern_N])
        self.encoder_b2m = weight_variable([self.Latent_layern_N])
        self.encoder_w2v = weight_variable([self.hLayer_N, self.Latent_layern_N])
        self.encoder_b2v = weight_variable([self.Latent_layern_N])

        self.decoder_w1 = weight_variable([self.Latent_layern_N, self.hLayer_N])
        self.decoder_b1 = weight_variable([self.hLayer_N])
        self.decoder_w2 = weight_variable([self.hLayer_N, 784])
        self.decoder_b2 = weight_variable([784])

        #Encoder Network
        h1 = tf.nn.tanh(tf.add(tf.matmul(self.inputs, self.encoder_w1), self.encoder_b1))

        self.mean = tf.add(tf.matmul(h1, self.encoder_w2m), self.encoder_b2m)

        self.log_var = tf.add(tf.matmul(h1, self.encoder_w2v), self.encoder_b2v)

        epsilon = tf.random.normal(tf.shape(self.log_var), dtype=tf.float32, mean=0.0, stddev=1.0)

        # Reparametrization step

        self.latent_layer = self.mean + tf.exp(0.5*self.log_var)*epsilon

        # Decoder Network

        h2 = tf.nn.tanh(tf.add(tf.matmul(self.latent_layer, self.decoder_w1), self.decoder_b1))

        self.out = tf.nn.sigmoid(tf.add(tf.matmul(h2, self.decoder_w2), self.decoder_b2))

        # Decoder network for testing

        decoder_h2 = tf.nn.tanh(tf.add(tf.matmul(self.noise_inputs, self.decoder_w1), self.decoder_b1))

        self.decoded_out = tf.nn.sigmoid(tf.add(tf.matmul(decoder_h2, self.decoder_w2), self.decoder_b2))

    def _loss(self):

        # Reproduction loss
        data_fidelity_loss = self.inputs*tf.log(1e-10+self.out)+ (1-self.inputs)*tf.log(1e-10+1-self.out)
        data_fidelity_loss = -tf.reduce_sum(data_fidelity_loss,1)

        # KL divergence loss between decoder net and prior on latent variables
        kl_div_loss = 1 + self.log_var - tf.square(self.mean) - tf.exp(self.log_var)
        kl_div_loss = -0.5*tf.reduce_sum(kl_div_loss,1)

        alpha = 1
        beta = 1
        # Loss missing  Reproduction loss + KL div loss
        self.loss = tf.reduce_mean(alpha*data_fidelity_loss + beta*kl_div_loss)

    def train(self, inputs):
        _, loss = self.sess.run([self.optimizer, self.loss], feed_dict={self.inputs:inputs})
        return loss
    
    def decode(self, noise):
        output = self.sess.run(self.decoded_out, feed_dict={self.noise_inputs:noise})
        return output




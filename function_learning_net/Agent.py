import numpy as np 
import tensorflow as tf

def weight_initializer(shape, var_name):
    init_val = tf.truncated_normal(shape)
    return tf.Variable(init_val, name=var_name)

def bias_initializer(shape, var_name):
    init_val = tf.constant(0.01,shape=shape)
    return tf.Variable(init_val, name=var_name)

class net(object):

    def __init__(self, sess, lr, obs_dim, output_dim):
        self.sess = sess
        self.lr = lr
        self.obs_dim = obs_dim
        self.output_dim = output_dim
        self.NN_hidden_layer1 = 100
        self.NN_hidden_layer2 = 50

        self._placeholder()
        self.out = self._init_net()

        self.loss = tf.reduce_mean((self.output_ph-self.out)**2)

        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def _placeholder(self):
        self.inputs_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim], name='Inputs')
        self.output_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.output_dim], name='Output')
               

    def _init_net(self):
        w1 = weight_initializer([self.obs_dim, self.NN_hidden_layer1], 'W1')
        b1 = bias_initializer([self.NN_hidden_layer1], 'b1')
        w2 = weight_initializer([self.NN_hidden_layer1, self.NN_hidden_layer2], 'W2')
        b2 = bias_initializer([self.NN_hidden_layer2],'b2')
        w3 = weight_initializer([self.NN_hidden_layer2, self.output_dim],'W3')
        b3 = bias_initializer([self.output_dim], 'b3')

        h1 = tf.nn.tanh(tf.matmul(self.inputs_ph, w1)+b1)
        h2 = tf.nn.tanh(tf.matmul(h1, w2)+b2)

        out = tf.matmul(h2, w3)+b3

        return out

    def train(self, inputs, outputs):
        self.sess.run(self.optimize, feed_dict={self.inputs_ph:inputs, self.output_ph:outputs})
        return self.sess.run(self.loss, feed_dict={self.inputs_ph:inputs, self.output_ph:outputs})

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={self.inputs_ph:inputs})
        
        


import numpy as np
import tensorflow as tf

from Agent import net 
import matplotlib.pyplot as plt

def generate_data(N):
    x = np.linspace(-np.pi, np.pi, N)
    y = np.sin(x)+np.random.normal(loc=0.0, scale=0.1,size=N)

    return x,y

def main():
    data_size = 100
    x_target,y_target = generate_data(data_size)

    with tf.Session() as sess:
        agent = net(sess, 0.001, 1,1)
        sess.run(tf.global_variables_initializer())
        loss_hist = []
        for steps in range(5000):
            loss = agent.train(np.reshape(x_target,(data_size,1)),np.reshape(y_target,(data_size,1)))
            loss_hist.append(loss)

        y = agent.predict(np.reshape(x_target,(data_size,1)))

    plt.figure(1)
    plt.plot(x_target,y_target)
    plt.plot(x_target,y)
    plt.xlabel('X')
    plt.ylabel('Y=f(X)')
    plt.grid()
    plt.show()

    plt.figure(2)
    plt.plot(loss_hist)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.show()


if __name__=='__main__':
    main()
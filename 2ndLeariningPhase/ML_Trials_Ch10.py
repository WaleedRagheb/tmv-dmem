# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:18:10 2018

@author: HP
"""

# Load the Pima Indians diabetes dataset from CSV URL
import numpy as np
import tensorflow as tf
#import urllib
## URL for the Pima Indians Diabetes dataset (UCI Machine Learning Repository)
#url = "http://goo.gl/j0Rvxq"
## download the file
#raw_data = urllib.urlopen(url)
## load the CSV file as a numpy matrix
#dataset = np.loadtxt(raw_data, delimiter=",")
#print(dataset.shape)
## separate the data from the target attributes
#X = dataset[:,0:7]
#y = dataset[:,8]

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 10, 1, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out



x_train = tf.cast(np.genfromtxt(r"./MLResults/Train_Test/TrainingData.csv", delimiter=",", usecols=[0,1,2,3,4,5,6,7,8,9]),tf.float32)
y_train  = tf.cast(np.genfromtxt(r"./MLResults/Train_Test/TrainingData.csv", delimiter=",", usecols=[10], dtype=np.str), tf.float32)

x_test = np.genfromtxt(r"./MLResults/Train_Test/TestingData.csv", delimiter=",", usecols=[0,1,2,3,4,5,6,7,8,9])
y_test  = np.genfromtxt(r"./MLResults/Train_Test/TestingData.csv", delimiter=",", usecols=[10], dtype=np.str)



keep_prob = tf.placeholder(tf.float32)


learning_rate = 0.001
training_epochs = 500
bat_size = 128
Ds_step = 10

# NN training Parameters
num_input = 10 # MNIST data 28 x 28
num_classes = 2 # MNIST classes 0 -> 9
dropout = 0.75 # Dropout, probability to keep units


# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}



# Construct model
logits = conv_net(x_train, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=y_train))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_train, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
X = x_train
Y = y_train
 # Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_epochs+1):
#        batch_x, batch_y = mnist.train.next_batch(bat_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: x_train, Y: y_train, keep_prob: dropout})
        if step % Ds_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: x_train,
                                                                 Y: y_train,
                                                                 keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: x_test,
                                      Y: y_test,
                                      keep_prob: 1.0}))



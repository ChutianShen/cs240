from __future__ import print_function

import sys
import os
import os.path
import math
import argparse
import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Uncomment when training on only CPU
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

parser = argparse.ArgumentParser()
parser.add_argument('--mode', help='LINEAR/RELU/MAXOUT')

parser.add_argument('--dropout', type=float, default=1, help='Set the dropout. 1: open dropout 2: close dropout')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 50]')
parser.add_argument('--display_step', type=int, default=1, help='The frequency to displace training progress')
parser.add_argument('--batch_size', type=int, default=100, help='Batch Size during training [default: 24]')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]') # No need in this simple case
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')             # No need in this simple case
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]') # No need in this simple case
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]') # No need in this simple case
FLAGS = parser.parse_args()

MODE = FLAGS.mode
DROPOUT = FLAGS.dropout
DISPLAY_STEP = FLAGS.display_step
BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate


def create_weight_variable(name, shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    return tf.Variable(initializer(shape=shape), name=name)


def create_bias_variable(name, shape):
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name=name)


tf.logging.set_verbosity(tf.logging.ERROR)
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

# Parameters
learning_rate = BASE_LEARNING_RATE
training_epochs = MAX_EPOCH
batch_size = BATCH_SIZE
display_step = DISPLAY_STEP
logs_path = '/tmp/tensorflow_logs/example'
keep_prob = tf.placeholder("float")

# tf Graph Input
# mnist data image of shape 28*28=784
x = tf.placeholder(tf.float32, [None, 784], name='InputData')
# 0-9 digits recognition => 10 classes
y = tf.placeholder(tf.float32, [None, 10], name='LabelData')


def linear():
    W1 = create_weight_variable('Weights', [784, 10])
    b1 = create_bias_variable('Bias', [10])
    return tf.nn.softmax(tf.matmul(x, W1) + b1)


def hidden_relu():
    W1 = create_weight_variable('Weights', [784, 100])
    b1 = create_bias_variable('Bias', [100])

    W2 = create_weight_variable('Weights2', [100, 10])
    b2 = create_bias_variable('Bias2', [10])
    t = tf.nn.relu(tf.matmul(x, W1) + b1)
    t_drop = tf.nn.dropout(t, keep_prob)
    return tf.nn.softmax(tf.matmul(t_drop, W2) + b2)


def hidden_maxout():
    W1 = create_weight_variable('Weights', [784, 100])
    b1 = create_bias_variable('Bias', [100])

    W2 = create_weight_variable('Weights2', [50, 10])
    b2 = create_bias_variable('Bias2', [10])

    from maxout import max_out
    t = max_out(tf.matmul(x, W1) + b1, 50)
    t_drop = tf.nn.dropout(t, keep_prob)
    return tf.nn.softmax(tf.matmul(t_drop, W2) + b2)


def select_model():
    usage = 'Usage: python mnist_maxout_example.py (LINEAR|RELU|MAXOUT)'
    # assert len(sys.argv) == 2, usage
    # t = sys.argv[1].upper()
    print('Type = ' + MODE)
    if MODE == 'LINEAR':
        return linear()
    elif MODE == 'RELU':
        return hidden_relu()
    elif MODE == 'MAXOUT':
        return hidden_maxout()
    else:
        raise Exception('Unknown type. ' + usage)



# Construct model and encapsulating all ops into scopes, making
# Tensorboard's Graph visualization more convenient
with tf.name_scope('Model'):
    # Model
    pred = select_model()
with tf.name_scope('Loss'):
    # Minimize error using cross entropy
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
with tf.name_scope('SGD'):
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
with tf.name_scope('Accuracy'):
    # Accuracy
    acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Create a summary to monitor cost tensor
tf.summary.scalar('loss', cost)
# Create a summary to monitor accuracy tensor
tf.summary.scalar('accuracy', acc)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    start_time = time.time()
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop), cost op (to get loss value)
            # and summary nodes
            _, c, summary = sess.run([optimizer, cost, merged_summary_op],
                                     feed_dict={x: batch_xs, y: batch_ys, keep_prob: DROPOUT})

            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))

    end_time = time.time()

    print('Optimization Finished!')
    print("Time Consuming: " + str(end_time - start_time))

    # Test model
    # Calculate accuracy
    print('Accuracy:', acc.eval({x: mnist.test.images, y: mnist.test.labels, keep_prob: 1}))

    print('Run the command line:\n' \
          '--> tensorboard --logdir=/tmp/tensorflow_logs ' \
          '\nThen open http://0.0.0.0:6006/ into your web browser')

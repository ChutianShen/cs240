import os
import os.path
import math
import time
import numpy as np
import tensorflow as tf

import os
import argparse
from include.data import get_data_set

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default= 'RELU', help='RELU/MAXOUT')
parser.add_argument('--dropout', type=float, default=0, help='Set the dropout')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--max_epoch', type=int, default=25000, help='Epoch to run')
parser.add_argument('--display_step', type=int, default=1, help='The frequency to displace training progress')
parser.add_argument('--batch_size', type=int, default=128, help='Batch Size during training [default: 24]')
parser.add_argument('--learning_rate', type=float, default=0.04, help='Initial learning rate [default: 0.05]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--job_name', type=str, default="", help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--task_index', type=int, default=0, help='Decay rate for lr decay [default: 0.5]')
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
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
ITERATION = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
NUM_EPOCHS_PER_DECAY = 350.0

if MODE != 'RELU' and MODE != 'MAXOUT':
    print ("Please input mode: RELU or MAXOUT !!!")

# BATCH_SIZE = 128
# learning_rate = 0.05
# MAX_STEP = 10000 # with this setting, it took less than 30 mins on my laptop to train.


def inference(images):
    '''
    Args:
        images: 4D tensor [batch_size, img_width, img_height, img_channel]
    Notes:
        In each conv layer, the kernel size is:
        [kernel_size, kernel_size, number of input channels, number of output channels].
        number of input channels are from previuous layer, if previous layer is THE input
        layer, number of input channels should be image's channels.
        
            
    '''
    #conv1, [5, 5, 3, 96], The first two dimensions are the patch size,
    #the next is the number of input channels, 
    #the last is the number of output channels
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights', 
                                  shape = [3, 3, 3, 96],
                                  dtype = tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.05,dtype=tf.float32)) 
        biases = tf.get_variable('biases', 
                                 shape=[96],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        if MODE == 'RELU':
            conv1 = tf.nn.relu(pre_activation, name= scope.name)
        elif MODE == 'MAXOUT':
            conv1 = tf.contrib.layers.maxout(pre_activation, 96, name=scope.name)
            # conv1 = max_out(pre_activation, 50, name=scope.name)

        '''
        keep_prob_conv1 = 0.2
        if DROPOUT == 1:
            conv1 = tf.nn.dropout(conv1, keep_prob_conv1)
        '''
    
    #pool1 and norm1   
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1],strides=[1,2,2,1],
                               padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm1')
        '''
	keep_prob1 = 0.25
        if DROPOUT == 1:
            norm1 = tf.nn.dropout(norm1, keep_prob1) 
        '''
    
    #conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,96, 64],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.05,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[64], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        if MODE == 'RELU':
            conv2 = tf.nn.relu(pre_activation, name='conv2')
        elif MODE == 'MAXOUT':
            conv2 = tf.contrib.layers.maxout(pre_activation, 32, name=scope.name)
            # conv2 = max_out(pre_activation, 50, name=scope.name)
        '''
        keep_prob_conv2 = 0.2
        if DROPOUT == 1:
            conv1 = tf.nn.dropout(conv2, keep_prob_conv2) 
        '''

    #pool2 and norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75,name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,1,1,1],
                               padding='SAME',name='pooling2')

        '''
        keep_prob2 = 0.25
        if DROPOUT == 1:
            pool2 = tf.nn.dropout(pool2, keep_prob2)
        '''
    #local3
    with tf.variable_scope('local3') as scope:
        dim = pool2.shape[1].value * pool2.shape[2].value * pool2.shape[3].value
        weights = tf.get_variable('weights',
                                  shape=[dim,384],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.004,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[384],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        '''
        keep_prob_local3 = 0.5
        if DROPOUT == 1:
            reshape = tf.nn.dropout(reshape, keep_prob_local3)
        '''
        flatten = tf.reshape(pool2, (-1, dim))
        local3 = tf.nn.relu(tf.matmul(flatten, weights) + biases, name=scope.name)
        

        if MODE == 'RELU':
            local3 = tf.nn.relu(tf.matmul(flatten, weights) + biases, name=scope.name)
        elif MODE == 'MAXOUT':
            local3 = tf.contrib.layers.maxout(tf.matmul(flatten, weights) + biases, 384, name=scope.name)
            # local3 = max_out(pre_activation, 50, name=scope.name)

    
    #local4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[384,192],
                                  dtype=tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.004,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[192],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        keep_prob_local4 = 0.5
        if DROPOUT == 1:
            local3 = tf.nn.dropout(local3, keep_prob_local4)


        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')
        
        '''
        if MODE == 'RELU':
            local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name = 'scope.name')
        elif MODE == 'MAXOUT':
            local4 = tf.contrib.layers.maxout(tf.matmul(local3, weights) + biases, 192, name='local4')
            # local4 = max_out(pre_activation, 50, name=scope.name)
        '''
        
    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[192, 10],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.004,dtype=tf.float32))
        biases = tf.get_variable('biases', 
                                 shape=[10],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        '''        
        keep_prob_softmax = 0.5
        if DROPOUT == 1:
            local4 = tf.nn.dropout(local4, keep_prob_softmax)
        '''
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')
    
    return softmax_linear

def losses(logits, labels):
    labels = tf.cast(labels, tf.int64)

    # to use this loss fuction, one-hot encoding is needed!
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy_per_example')

    loss = tf.reduce_mean(cross_entropy, name='loss')
    tf.summary.scalar('loss', loss)
    return loss

def asynchrounous_train():
    parameter_servers = ["127.0.0.1:2222"]
    workers = ["127.0.0.1:2223", "127.0.0.1:2224", "127.0.0.1:2225"]
    cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})
    log_dir = 'log'
    train_x, train_y, train_l = get_data_set()
    test_x, test_y, test_l = get_data_set("test")
    start_time = time.time()

    print "Loading cifar10 images"
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    print "Successfully building the server"
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):
            print "Enter the worker mode {}".format(FLAGS.task_index)

            my_global_step = tf.Variable(0, name='global_step', trainable=False)
            images = tf.placeholder(tf.float32, [None, 32, 32, 3])
            labels = tf.placeholder(tf.float32, [None, train_y.shape[1]])
            logits = inference(images)
            loss = losses(logits, labels)
            acc = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            acc = tf.reduce_mean(tf.cast(acc, tf.float32))
            global_step = tf.Variable(0, trainable=True)

            num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
            decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
            lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps, 0.9, staircase=True)

            optimizer = tf.train.GradientDescentOptimizer(lr)
            train_op = optimizer.minimize(loss, global_step=my_global_step)

            saver = tf.train.Saver(tf.global_variables())
            tf.summary.scalar("loss", loss)
            summary_op = tf.summary.merge_all()

            init = tf.global_variables_initializer()

            print("Variables initialized ...")

        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0), global_step=global_step, init_op=init)

        with sv.prepare_or_wait_for_session(server.target) as sess:
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
            for epoch in np.arange(MAX_EPOCH):
                for _ in np.arange(ITERATION):
                    randidx = np.random.randint(len(train_x), size=FLAGS.batch_size)
                    batch_xs = train_x[randidx].reshape((-1, 32, 32, 3))
                    batch_ys = train_y[randidx].astype("float32")

                    _, loss_value, summary_str, step = sess.run([train_op, loss, summary_op, my_global_step],
                                                                feed_dict={images: batch_xs, labels: batch_ys})
                    summary_writer.add_summary(summary_str, step)

                    if step % 100 == 0 and step > 0:
                        print ('Task: %d, Step: %d, loss: %.4f' % (FLAGS.task_index, step, loss_value))

                    if step % 1000 == 0 and step > 0 and FLAGS.task_index != 1:
                        batch_xs = test_x.reshape((-1, 32, 32, 3))
                        batch_ys = test_y.astype("float32")
                        accuracy = sess.run(acc, feed_dict={images: batch_xs, labels: batch_ys})
                        print ('Elapsed time: %.4f, Accuracy: %.4f' % (time.time() - start_time, accuracy))
                        checkpoint_path = os.path.join(log_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)

def train():
    my_global_step = tf.Variable(0, name='global_step', trainable=False)

    log_dir = 'log'

    train_x, train_y, train_l = get_data_set()
    test_x, test_y, test_l = get_data_set("test")

    images = tf.placeholder(tf.float32, [None, 32, 32, 3])
    labels = tf.placeholder(tf.float32, [None, train_y.shape[1]])
    logits = inference(images)
    #y_pred_cls = tf.argmax(logits, 10)
    loss = losses(logits, labels)
    acc = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

    global_step = tf.Variable(0, trainable=True)
    learning_rate = tf.train.exponential_decay(BASE_LEARNING_RATE, global_step=global_step, decay_steps=10, decay_rate=0.9)   
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=my_global_step)
    
    saver = tf.train.Saver(tf.global_variables())
    tf.summary.scalar("loss", loss)
    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    sess = tf.Session()

    sess.run(init)

    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    for epoch in np.arange(5):
        for step in np.arange(ITERATION):
            randidx = np.random.randint(len(train_x), size=FLAGS.batch_size)
            batch_xs = train_x[randidx].reshape((-1, 32, 32, 3))
            batch_ys = train_y[randidx].astype("float32")

            _, loss_value, summary_str = sess.run([train_op, loss, summary_op], feed_dict={images: batch_xs, labels: batch_ys})
            summary_writer.add_summary(summary_str, step)

            if step % 50 == 0 and step > 0:
                print ('Step: %d, loss: %.4f' % (step, loss_value))

            if step % 1000 == 0 and step > 0 and FLAGS.task_index != 1:
                batch_xs = test_x.reshape((-1, 32, 32, 3))
                batch_ys = test_y.astype("float32")
                accuracy = sess.run(acc, feed_dict={images: batch_xs, labels: batch_ys})
                print ('Accuracy: %.4f' % accuracy)
                checkpoint_path = os.path.join(log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


if __name__ == "__main__":
    asynchrounous_train()

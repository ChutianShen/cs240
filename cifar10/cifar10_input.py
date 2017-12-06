import tensorflow as tf
import numpy as np
import os
import sys
# reload(sys)
# sys.setdefaultencoding('utf8')

# Reading data

def read_cifar10(data_dir, is_train, batch_size, shuffle):
    """Read CIFAR10
    
    Args:
        data_dir: the directory of CIFAR10
        is_train: boolen
        batch_size:
        shuffle:       
    Returns:
        label: 1D tensor, tf.int32
        image: 4D tensor, [batch_size, height, width, 3], tf.float32
    
    """
    img_width = 32
    img_height = 32
    img_depth = 3
    label_bytes = 1 # 笔记中有写，第一位是label
    image_bytes = img_width*img_height*img_depth
    
    
    with tf.name_scope('input'):    # 在tensorboard中，这个就叫做input。画出来比较好看
        
        if is_train:
            filenames = [os.path.join(data_dir, 'data_batch_%d.bin' %ii)
                                        for ii in np.arange(1, 6)]
        else:
            filenames = [os.path.join(data_dir, 'test_batch.bin')]
          
        filename_queue = tf.train.string_input_producer(filenames)  # 不同于slice_input_producer. 对于slice，输入是一个list，分别是label和数据。而这一个都在一个二进制文件
    
        reader = tf.FixedLengthRecordReader(label_bytes + image_bytes)  # 每一次要读的长度。3073就是一张图片
    
        key, value = reader.read(filename_queue)    # 用reader读文件，得到一个key和对应的value
           
        record_bytes = tf.decode_raw(value, tf.uint8)
        
        label = tf.slice(record_bytes, [0], [label_bytes])   
        label = tf.cast(label, tf.int32)    # 把label的数据转换成int

        print (label)
        
        image_raw = tf.slice(record_bytes, [label_bytes], [image_bytes])    # slice都是从固定长度的list中取值
        image_raw = tf.reshape(image_raw, [img_depth, img_height, img_width])      # 32 * 32 * 3
        image = tf.transpose(image_raw, (1,2,0)) # convert from D/H/W to H/W/D      # 转换一下顺序。 把channel放到最后一位
        image = tf.cast(image, tf.float32)  # 转换格式

     
#        # data argumentation

#        image = tf.random_crop(image, [24, 24, 3])# randomly crop the image size to 24 x 24
#        image = tf.image.random_flip_left_right(image)
#        image = tf.image.random_brightness(image, max_delta=63)
#        image = tf.image.random_contrast(image,lower=0.2,upper=1.8)


        
        image = tf.image.per_image_standardization(image) # substract off the mean and divide by the variance 标准化


        if shuffle:
            images, label_batch = tf.train.shuffle_batch(
                                    [image, label], 
                                    batch_size = batch_size,
                                    num_threads= 16,
                                    capacity = 2000,    # 最多一次从队列中读取的个数
                                    min_after_dequeue = 1500) # 每次从队列中取走一个batch，剩下的最少的个数
        else:
            images, label_batch = tf.train.batch(
                                    [image, label],
                                    batch_size = batch_size,
                                    num_threads = 16,
                                    capacity= 2000)

        
        return images, tf.reshape(label_batch, [batch_size])






## ONE-HOT      
        # n_classes = 10
        # label_batch = tf.one_hot(label_batch, depth= n_classes)
        #
        # print (label_batch)
        #
        # return images, tf.reshape(label_batch, [batch_size, n_classes])
    




#%%   TEST
# To test the generated batches of images
# When training the model, DO comment the following codes



# import matplotlib.pyplot as plt
#
# data_dir = '/Users/kevin_sct/deepLearningStudy/useSelfData/My_TensorFlow_tutorials/02_CIFAR10/data'
# BATCH_SIZE = 10
# image_batch, label_batch = read_cifar10(data_dir,
#                                        is_train=True,
#                                        batch_size=BATCH_SIZE,
#                                        shuffle=True)
#
# with tf.Session() as sess:
#    i = 0
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#
#    try:
#        while not coord.should_stop() and i<1:
#
#            img, label = sess.run([image_batch, label_batch])
#
#            # just test one batch
#            for j in np.arange(BATCH_SIZE):
#                print('label:  ' + str(label[j]))
#                plt.imshow(img[j,:,:,:])
#                plt.show()
#            i+=1
#
#    except tf.errors.OutOfRangeError:
#        print('done!')
#    finally:
#        coord.request_stop()
#    coord.join(threads)

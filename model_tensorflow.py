
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import time
import os
from dataset import input_fn
import cv2
import datetime
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.datetime.now())
train_log_dir = 'weights/logs/' + TIMESTAMP
class Gan():

    def __init__(self,data_shape,z_d, learning_rate,batch_size):

        self.data_shape = data_shape
        self.length = self.data_shape[0] * self.data_shape[1] * self.data_shape[2]
        self.label_dim = 100
        self.z_d = z_d
        self.d_learning_rate = learning_rate
        self.g_learning_rate = 0.0001
        self.batch_size = batch_size
        self.beta1 = 0.5
        self.build_net()

    def Generator(self, z, label, is_training, reuse):
        depths = [1024, 512, 256, 128, 64] + [self.data_shape[2]]
        with tf.variable_scope("Generator", reuse = reuse):
            with tf.variable_scope("g_input",reuse=reuse):

                input_all = tf.concat([z, label],axis = -1)

            with tf.variable_scope("g_fc1", reuse = reuse):
                output = tf.layers.dense(input_all, depths[0]*4*4, trainable = is_training)
                output = tf.reshape(output, [self.batch_size, 4, 4, depths[0]])
                #output = tf.nn.relu(tf.layers.batch_normalization(output, training = is_training))
                output = tf.nn.relu(tf.layers.batch_normalization(output, training = is_training))

            with tf.variable_scope("g_dc1", reuse = reuse):
                output = tf.layers.conv2d_transpose(output, depths[1], [5,5], strides =(2,2), padding ="SAME", trainable = is_training, name="conv2d_transpose")
             #   output = tf.layers.dropout(output,training = is_training)
                output = tf.nn.relu(tf.layers.batch_normalization(output, training = is_training))

            with tf.variable_scope("g_dc2", reuse = reuse):
                output = tf.layers.conv2d_transpose(output, depths[2], [5,5], strides = (2,2), padding ="SAME", trainable = is_training, name="conv2d_transpose")
             #   output = tf.layers.dropout(output, training=is_training)
                output = tf.nn.relu(tf.layers.batch_normalization(output, training = is_training))

            with tf.variable_scope("g_dc3", reuse = reuse):
                output = tf.layers.conv2d_transpose(output,depths[3], [5,5], strides = (2,2), padding ="SAME", trainable = is_training, name="conv2d_transpose")
              #  output = tf.layers.dropout(output, training=is_training)
                output = tf.nn.relu(tf.layers.batch_normalization(output, training = is_training))

            with tf.variable_scope("g_dc4", reuse = reuse):
                output = tf.layers.conv2d_transpose(output,depths[4], [5,5], strides = (2,2), padding = "SAME", trainable = is_training, name="conv2d_transpose")
              #  output = tf.layers.dropout(output, training=is_training)
                output = tf.nn.relu(tf.layers.batch_normalization(output, training = is_training))

            with tf.variable_scope("g_dc5", reuse = reuse):
                output = tf.layers.conv2d_transpose(output, depths[5], [5, 5], strides=(2, 2), padding="SAME",  trainable = is_training, name="conv2d_transpose")
                result = tf.nn.tanh(output)


        return result

    def conv_cond_concat(self,x, y):
        """Concatenate conditioning vector on feature map axis."""
        x_shapes = x.get_shape()
        y_shapes = y.get_shape()
        return tf.concat([x, y * tf.ones([self.batch_size, x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

    # def Discriminator(self,X, label, is_training, reuse):
    #     depths = [self.data_shape[2]] + [64, 128, 256, 512, 1024]
    #     with tf.variable_scope("Discriminator", reuse = reuse):
    #
    #         with tf.variable_scope("d_input",reuse= reuse):
    #
    #             label = tf.reshape(label,(-1, 1, 1, self.label_dim))
    #
    #             input_all = self.conv_cond_concat(X, label)
    #
    #         with tf.variable_scope("d_cv1", reuse = reuse):
    #             output = tf.layers.conv2d(input_all, depths[1], [5,5], strides = (2,2), padding ="SAME", trainable = is_training)
    #             output = tf.nn.leaky_relu(tf.layers.batch_normalization(output, training = is_training))
    #
    #         with tf.variable_scope("d_cv2", reuse = reuse):
    #             output = tf.layers.conv2d(output, depths[2], [5,5], strides = (2,2), padding ="SAME", trainable = is_training)
    #             output = tf.nn.leaky_relu(tf.layers.batch_normalization(output, training = is_training))
    #
    #         with tf.variable_scope("d_cv3", reuse = reuse):
    #             output = tf.layers.conv2d(output, depths[3], [5,5], strides = (2,2), padding = "SAME", trainable = is_training)
    #             output = tf.nn.leaky_relu(tf.layers.batch_normalization(output, training = is_training))
    #
    #         with tf.variable_scope("d_cv4", reuse = reuse):
    #             output = tf.layers.conv2d(output, depths[4], [5,5], strides = (2,2), padding ="SAME", trainable = is_training)
    #           #  output = tf.nn.leaky_relu(tf.layers.batch_normalization(output, training = is_training))
    #             output = tf.nn.leaky_relu(output)
    #
    #         with tf.variable_scope("d_cv5", reuse = reuse):
    #             d_no_sigmoid_logits = tf.layers.conv2d(output, 1, kernel_size=[5,5], strides=(1,1), padding = "SAME", trainable = is_training)
    #             d_logits = tf.nn.sigmoid(d_no_sigmoid_logits)
    #
    #
    #         return d_logits,d_no_sigmoid_logits
    #
    def Discriminator(self,X, label, is_training, reuse):
        depths = [self.data_shape[2]] + [64, 128, 256, 512, 1024]
        with tf.variable_scope("Discriminator", reuse = reuse):

            with tf.variable_scope("d_input",reuse= reuse):

                label = tf.reshape(label,(-1, 1, 1, self.label_dim))

                input_all = self.conv_cond_concat(X,label)

            with tf.variable_scope("d_cv1", reuse = reuse):
                output = tf.layers.conv2d(input_all, depths[1], [5,5], strides = (2,2), padding ="SAME", trainable = is_training)
                output = tf.nn.leaky_relu(output)

            with tf.variable_scope("d_cv2", reuse = reuse):
                output = tf.layers.conv2d(output, depths[2], [5,5], strides = (2,2), padding ="SAME", trainable = is_training)
                output = tf.nn.leaky_relu(output)

            with tf.variable_scope("d_cv3", reuse = reuse):
                output = tf.layers.conv2d(output, depths[3], [5,5], strides = (2,2), padding = "SAME", trainable = is_training)
                output = tf.nn.leaky_relu(output)

            with tf.variable_scope("d_cv4", reuse = reuse):
                output = tf.layers.conv2d(output, depths[4], [5,5], strides = (2,2), padding ="SAME", trainable = is_training)
                output = tf.nn.leaky_relu(output)

            with tf.variable_scope("d_cv5", reuse = reuse):
                output = tf.layers.conv2d(output, depths[5], [5,5], strides = (2,2), padding ="SAME", trainable = is_training)
                # output = tf.nn.leaky_relu(tf.layers.batch_normalization(output, training = is_training))
                output = tf.nn.leaky_relu(output)


            with tf.variable_scope("d_fc1", reuse = reuse):
                output = tf.layers.flatten(output)
                d_no_sigmoid_logits = tf.layers.dense(output,1, trainable= is_training)
                d_logits = tf.nn.sigmoid(d_no_sigmoid_logits)

            return d_logits,d_no_sigmoid_logits

    def plot_and_save(self, order, images):
        os.makedirs("images",exist_ok=True)
        batch_size = len(images)
        n = np.int(np.sqrt(batch_size))
        image_size = np.shape(images)[2]
        n_channel = np.shape(images)[3]
        images = np.reshape(images, [-1,image_size,image_size,n_channel])
        canvas = np.empty((n * image_size, n * image_size))
        for i in range(n):
            for j in range(n):
                canvas[i*image_size: (i + 1) * image_size , j * image_size:(j + 1) * image_size] = \
                                  images[n * i + j].reshape(self.data_shape[0],self.data_shape[1])
        plt.figure(figsize =(8,8))
        plt.imshow(canvas, cmap ="gray")
        label = "Epoch: {0}".format(order+1)
        plt.xlabel(label)

        if type(order) is str:
            file_name = order
        else:
            file_name = os.path.join("images","Mnist_canvas" + str(order))

        plt.savefig(file_name)
        print(os.getcwd())
        print("Image saved in file: ", file_name)
        plt.close()

    def build_net(self):
        self.X = tf.placeholder(tf.float32 , shape = [None, self.length], name ="Input_data")
        self.X_img = tf.reshape(self.X, [-1] + self.data_shape)
        self.z = tf.placeholder(tf.float32, shape = [None, self.z_d], name ="latent_var")
        self.c = tf.placeholder(tf.int32,shape=[None,],name = "label")
        self.C = tf.one_hot(self.c,depth=self.label_dim)

        self.G = self.Generator(self.z,self.C ,is_training = True, reuse = False)

        D_fake_logits,self.D_fake_logits = self.Discriminator(self.G,self.C, is_training = True, reuse = False)
        D_true_logits,self.D_true_logits = self.Discriminator(self.X_img, self.C,is_training = True, reuse = True)

        # self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self. D_fake_logits, labels = tf.ones_like(self.D_fake_logits)))
        # self.D_loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_true_logits , labels = tf.ones_like(self.D_true_logits)))
        # self.D_loss_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_fake_logits  , labels = tf.zeros_like(self.D_fake_logits)))
        self.D_loss = tf.reduce_mean(self.D_fake_logits) - tf.reduce_mean(self.D_true_logits)
        self.G_loss = - tf.reduce_mean(self.D_fake_logits)

        n_p_x = tf.reduce_sum(tf.cast(D_true_logits > 0.5, tf.int32))
        n_p_z = tf.reduce_sum(tf.cast(D_fake_logits < 0.5, tf.int32))
        self.D_acc = tf.divide(n_p_x + n_p_z, 2 * self.batch_size)

###     Gradient Penalty
        self.epsilon = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0.,maxval=1.)
        X_hat = self.X_img + self.epsilon * (self.G - self.X_img)
        D_X_sigmoid_hat, D_X_hat = self.Discriminator(X_hat, self.C,is_training = True, reuse = True)
        grad_D_X_hat = tf.gradients(D_X_hat, [X_hat])[0]

        red_idx = [x for x in range(1, X_hat.shape.ndims)]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_X_hat), reduction_indices=red_idx))
        self.gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2) * 10.0
        self.D_loss = self.D_loss + self.gradient_penalty

        total_vars = tf.trainable_variables()


        self.g_vars = [var for var in total_vars if any(x in var.name for x in ['Generator'])]
        self.d_vars = [var for var in total_vars if any(x in var.name for x in ['Discriminator'])]


        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            # self.g_optimization = tf.train.RMSPropOptimizer(learning_rate = self.g_learning_rate, decay = self.beta1).minimize(self.G_loss, var_list = self.g_vars)
            # self.d_optimization = tf.train.RMSPropOptimizer(learning_rate = self.d_learning_rate, decay = self.beta1).minimize(self.D_loss, var_list = self.d_vars)
            self.g_optimization = tf.train.AdamOptimizer(learning_rate = self.g_learning_rate, beta1 = self.beta1).\
                minimize(self.G_loss, var_list = self.g_vars)
            self.d_optimization = tf.train.AdamOptimizer(learning_rate = self.d_learning_rate, beta1 = self.beta1).\
                minimize(self.D_loss, var_list = self.d_vars)
        print("we successfully make the network")

    def training(self, epoch):
        start_time = time.time()
        sess = tf.Session()

        sess.run(tf.global_variables_initializer())

        dataset, data_num = input_fn(self.batch_size)
        iteration = dataset.make_initializable_iterator()
        dataset_input = iteration.get_next()
        sess.run(iteration.initializer)
        saver = tf.train.Saver()

        if os.listdir("./weights/"):
            ckpt = tf.train.get_checkpoint_state("./weights/")
            saver.restore(sess, ckpt.model_checkpoint_path)
        writer = tf.summary.FileWriter(train_log_dir, sess.graph)

        for i in range(epoch):
            total_batch = data_num
            d_value = 0
            g_value = 0
            labels = None

            for j in range((total_batch // self.batch_size)):
                gradient_penalty_all = 0
                d_acc_all = 0
                d_all = 0

                for k in range(5):
                    batch_xs, labels = sess.run(dataset_input)
                    batch_xs = np.reshape(batch_xs, (self.batch_size, self.length))
                    #     labels = np.reshape(labels,(self.batch_size,1))
                    z_sampled1 = np.random.uniform(low=-1.0, high=1.0, size=[self.batch_size, self.z_d])
                    Op_d, d_, d_acc, gradient_penalty = sess.run(
                        [self.d_optimization, self.D_loss, self.D_acc, self.gradient_penalty],
                        feed_dict={self.X: batch_xs, self.z: z_sampled1, self.c: labels})
                    gradient_penalty_all = gradient_penalty_all + gradient_penalty
                    d_acc_all = d_acc_all + d_acc
                    d_all = d_all + d_
                    print("%d,%d:%d,d_loss:%f,d_acc:%f,gradient_penalty:%f" % (i, j, total_batch // self.batch_size, d_,d_acc, gradient_penalty))

                z_sampled2 = np.random.uniform(low=-1.0, high=1.0, size=[self.batch_size, self.z_d])
                Op_g, g_ = sess.run([self.g_optimization, self.G_loss],
                                    feed_dict={self.z: z_sampled2, self.c: labels})
              #  print("%d,%d:%d,g_loss:%f" % (i, j, total_batch // self.batch_size, g_))

                self.images_generated = sess.run(self.G, feed_dict={self.z: z_sampled2, self.c: labels})
                print("%d,%d:%d,g_loss:%f  d_loss:%f  d_acc:%f  gradient_penalty: %f" % (
                i, j, total_batch // self.batch_size, g_, d_all/5, d_acc_all/5, gradient_penalty_all/5))

                s1 = tf.Summary(value=[tf.Summary.Value(tag="d_loss", simple_value=d_all/5)])
                s2 = tf.Summary(value=[tf.Summary.Value(tag="d_acc", simple_value=d_acc_all/5)])
                s3 = tf.Summary(value=[tf.Summary.Value(tag="g_loss", simple_value=g_)])
                s4 = tf.Summary(value=[tf.Summary.Value(tag="gradient_penalty", simple_value=gradient_penalty_all/5)])

                writer.add_summary(s1, j + i * int(total_batch / self.batch_size))
                writer.add_summary(s2, j + i * int(total_batch / self.batch_size))
                writer.add_summary(s3, j + i * int(total_batch / self.batch_size))
                writer.add_summary(s4, j + i * int(total_batch / self.batch_size))

                d_value += d_ / total_batch
                g_value += g_ / total_batch
                if j % 240 == 0:
                    saver.save(sess, "./weights/word-%d-ckpt" % i, global_step=j)
                    self.plot_and_save(i, self.images_generated)
            hour = int((time.time() - start_time) / 3600)
            min = int(((time.time() - start_time) - 3600 * hour) / 60)
            sec = int((time.time() - start_time) - 3600 * hour - 60 * min)
            print("Time: ", hour, "h", min, "min", sec, "sec", "   Epoch: ", i, "G_loss: ", g_value, "D_loss: ",
                  d_value)

    # def training(self, epoch):
    #     start_time = time.time()
    #     sess = tf.Session()
    #
    #     sess.run(tf.global_variables_initializer())
    #
    #
    #     dataset, data_num = input_fn(self.batch_size)
    #     iteration = dataset.make_initializable_iterator()
    #     dataset_input = iteration.get_next()
    #     sess.run(iteration.initializer)
    #     saver = tf.train.Saver()
    #
    #     if os.listdir("./weights/"):
    #         ckpt = tf.train.get_checkpoint_state("./weights/")
    #         saver.restore(sess, ckpt.model_checkpoint_path)
    #     writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    #
    #     for i in range(epoch):
    #         total_batch = data_num
    #         d_value = 0
    #         g_value = 0
    #         for j in range((total_batch//self.batch_size)):
    #             batch_xs,labels = sess.run(dataset_input)
    #             batch_xs = np.reshape(batch_xs,(self.batch_size,self.length))
    #        #     labels = np.reshape(labels,(self.batch_size,1))
    #             z_sampled1 = np.random.uniform(low = -1.0, high = 1.0, size = [self.batch_size, self.z_d])
    #             Op_d, d_,d_acc,gradient_penalty= sess.run([self.d_optimization, self.D_loss,self.D_acc,self.gradient_penalty], feed_dict = {self.X:batch_xs, self.z: z_sampled1, self.c:labels})
    #
    #             for k in range(5):
    #                 z_sampled2 = np.random.uniform(low = -1.0, high = 1.0, size = [self.batch_size, self.z_d])
    #                 Op_g, g_= sess.run([self.g_optimization, self.G_loss], feed_dict = {self.X:batch_xs, self.z: z_sampled2, self.c:labels})
    #                 print("%d,%d:%d,g_loss:%f" % (i, j, total_batch // self.batch_size, g_))
    #             self.images_generated = sess.run(self.G, feed_dict = {self.z:z_sampled2,self.c:labels})
    #             print("%d,%d:%d,g_loss:%f  d_loss:%f  d_acc:%f  gradient_penalty: %f"%(i,j,total_batch//self.batch_size,g_,d_,d_acc, gradient_penalty))
    #
    #             s1 = tf.Summary(value= [tf.Summary.Value(tag = "d_loss",simple_value = d_)])
    #             s2 = tf.Summary(value=[tf.Summary.Value(tag="d_acc", simple_value = d_acc)])
    #             s3 = tf.Summary(value=[tf.Summary.Value(tag="g_loss", simple_value = g_)])
    #             s4 = tf.Summary(value=[tf.Summary.Value(tag="gradient_penalty", simple_value=gradient_penalty)])
    #
    #             writer.add_summary(s1, j + i * int(total_batch / self.batch_size))
    #             writer.add_summary(s2, j + i * int(total_batch / self.batch_size))
    #             writer.add_summary(s3, j + i * int(total_batch / self.batch_size))
    #             writer.add_summary(s4, j + i * int(total_batch / self.batch_size))
    #
    #
    #             d_value += d_/total_batch
    #             g_value +=  g_/ total_batch
    #             if j % 240 ==0:
    #                saver.save(sess,"./weights/word-%d-ckpt"%i, global_step = j)
    #                self.plot_and_save(i, self.images_generated)
    #         hour = int((time.time() - start_time) / 3600)
    #         min = int(((time.time() - start_time) - 3600 * hour)/60)
    #         sec = int((time.time()  - start_time) - 3600 * hour - 60 * min)
    #         print("Time: ",hour,"h", min,"min",sec ,"sec","   Epoch: ", i, "G_loss: ", g_value, "D_loss: ",d_value)

    def testing(self):

            # sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state("./weights/")
            print(ckpt.model_checkpoint_path)

            saver = tf.train.Saver()
            batch_size = 20
            with tf.Session() as sess:
                saver.restore(sess, ckpt.model_checkpoint_path)
                for i in range(10000//batch_size):
                    label = i // (100//batch_size)
                    labels = np.ones(shape=(batch_size,)) * label
                    z_sampled_for_test = np.random.uniform(low=-1.0, high=1.0, size=[batch_size, self.z_d])
                    pic = sess.run(self.G, feed_dict={self.z: z_sampled_for_test,self.c:labels})
                    for j in range(self.batch_size):
                       sample = cv2.resize(pic[j],(128,128))
                       sample = (sample + 1) / 2 * 255
                       cv2.imwrite("result/%d.png"%(i*batch_size+j),sample)


                print("we have successfully completed the test ")



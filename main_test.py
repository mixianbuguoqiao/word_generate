import tensorflow as tf
# import data
import model_tensorflow as model
import os
from tensorflow.python.keras.layers import *


os.environ["CUDA_VISIBLE_DEVICES"]="0,2"
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("z_d", 100, "Dimension of z")
flags.DEFINE_float("learning_rate", 0.0001, "learning_rate")
flags.DEFINE_integer("batch_size", 20, "batch_size")
flags.DEFINE_integer("n_epoch", 1000, "number of epoch")

#train_data,train_labels,validation_data,validation_labels,test_data,test_labels = data.prepare_MNIST_Data()

network = model.Gan([128,128,1],FLAGS.z_d, FLAGS.learning_rate, FLAGS.batch_size)
#network.training(FLAGS.n_epoch)
network.testing()

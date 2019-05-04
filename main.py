import tensorflow as tf
# import data
import model_tensorflow as model
import os


os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("z_d", 100, "Dimension of z")
flags.DEFINE_float("learning_rate", 0.0001, "learning_rate")
flags.DEFINE_integer("batch_size", 20, "batch_size")
flags.DEFINE_integer("n_epoch", 1000, "number of epoch")

network = model.Gan([128,128,1],FLAGS.z_d, FLAGS.learning_rate, FLAGS.batch_size)
network.training(FLAGS.n_epoch)
#network.testing()

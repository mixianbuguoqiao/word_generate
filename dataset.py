import cv2
import tensorflow as tf
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
image_size = 128


def _read_py_function(filename, label):
  image_decoded = cv2.imread(filename.decode())
  image_decoded = cv2.cvtColor(image_decoded,cv2.COLOR_BGR2GRAY)
  image_decoded = cv2.resize(image_decoded, (image_size,image_size))
  image_decoded = np.reshape(image_decoded,[image_size,image_size,1])

  return image_decoded, label

def normalization(X):

  return (X-127.5) / 127.5

# Use standard TensorFlow operations to resize the images to a fixed shape.
def _resize_function(image_decoded, label):
  image_decoded.set_shape([None, None, None])
  image_resized = tf.image.resize_images(image_decoded, [image_size,image_size])
  image_resized = normalization(image_resized)
  return image_resized, label

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_label(filename):
    label = {}
    file_name = os.listdir(filename)
    for i in range(len(file_name)):
       label[i] = file_name[i]
    return label



def prepare_data(train_path):

    image_data = []
    image_label = []
    dict_label = get_label(train_path)


    for label in dict_label.keys():
        train_data = os.path.join(train_path,dict_label[label])
        for root, _, fnames in sorted(os.walk(train_data)):
            for fname in fnames:
                if is_image_file(fname):
                    path_data = os.path.join(train_data, fname)
                    image_data.append(path_data)
                    image_label.append(label)

    return  image_data,image_label

def input_fn(batch_size):


    dataset_filenames, dataset_labels = prepare_data("../train")

    data_nums = len(dataset_filenames)

    dataset = tf.data.Dataset.from_tensor_slices((dataset_filenames, dataset_labels))
    dataset = dataset.map(
        lambda filename, label: tuple(tf.py_func(
            _read_py_function, [filename, label], [tf.uint8, tf.int32])))
    dataset = dataset.map(_resize_function)
    dataset = dataset.shuffle(buffer_size=20000).batch(batch_size).repeat(1000)
    return dataset,data_nums
# dataset,data_num = input_fn(8)
# iteration =  dataset.make_initializable_iterator()
# dataset_input = iteration.get_next()
# sess = tf.Session()
# sess.run(iteration.initializer)
# data,label = sess.run(dataset_input)
# cv2.imwrite("1.jpg",(data[0]+1)*255.0)
# cv2.imwrite("2.jpg",(data[1]+1)*255.0)
# cv2.imwrite("3.jpg",(data[2]+1)*255.0)



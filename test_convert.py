import os
os.environ["KERAS_BACKEND"] = "jax"
import numpy as np
import tensorflow as tf
from keras.src import backend
from keras.src.utils import backend_utils

t = tf.constant(["Hello", "World"])
print(f"TF tensor: {t}")
converted = backend_utils.convert_tf_tensor(t)
print(f"Converted type: {type(converted)}")
print(f"Converted value: {converted}")

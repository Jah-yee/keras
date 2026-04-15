import os
import sys

backend = sys.argv[1] if len(sys.argv) > 1 else "jax"
os.environ["KERAS_BACKEND"] = backend

import re
import string
import keras
import numpy as np

strip_chars = string.punctuation
def my_standardize(input_string):
    print(f"Backend: {os.environ['KERAS_BACKEND']}")
    print(f"Type of input_string: {type(input_string)}")
    try:
        input_string = input_string.lower()
    except AttributeError:
        # If it's a tensor, we might need to handle it differently
        # But the user expects .lower() to work
        raise
    return re.sub(f"[{re.escape(strip_chars)}]", "", input_string)

layer = keras.layers.TextVectorization(standardize=my_standardize)
try:
    layer.adapt(["Hello, world."])
    print("Adapt successful")
except Exception as e:
    print(f"Caught exception: {e}")

import os
import os.path
import math
import time
import numpy as np
import tensorflow as tf

import cifar10

start_time = time.time()
cifar10.train()
end_time = time.time()
print("Time Consuming: " + str(end_time - start_time))
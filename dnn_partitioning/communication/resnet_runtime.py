#%% 
import pickle
import time
import numpy as np
from keras.models import load_model
from keras.datasets import cifar10

subtract_pixel_mean = True
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# m_exit1 = load_model("exit1.h5")


x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_test -= x_train_mean

#%% 
dic = {"name":[],"runtime":[]}
temp = x_test
for layer in m_exit1.layers:
    t0 = time.time()
    temp = layer(temp)
    t1 = time.time()
    dic["name"].append(layer.name)
    dic["runtime"].append((t1-t0)/len(x_test))

with open("runtime.pickle","wb") as f:
    pickle.dump(dic,f, pickle.HIGHEST_PROTOCOL)

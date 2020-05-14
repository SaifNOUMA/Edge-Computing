#%%
import zmq
import sys
import time
from fr_utils import *
from keras.models import load_model
from keras.utils import to_categorical
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import pickle

subtract_pixel_mean, num_classes = True, 10

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = to_categorical(y_train, num_classes)
# y_test = to_categorical(y_test, num_classes)

#%% --------------------------- LOADING THE MODEL ---------------------------
m_exit1 = load_model("exit1.h5")

#%% --------------------------- STARTING COMMUNICATION ---------------------------
edge_server , cloud, threshold = sys.argv[1:]
Threshold = float(threshold)
context = zmq.Context()

# Publish to the edge server
print("Connecting to Edge server…")
socket1 = context.socket(zmq.REQ)
socket1.connect(edge_server)   # "tcp://*:500"

# Publish to the cloud
print("Connecting to Cloud server…")
socket2 = context.socket(zmq.SUB)
socket2.setsockopt(zmq.SUBSCRIBE, b"")
socket2.connect(cloud)   # "tcp://*:500"

Thresholds = [10**(-x) for x in range(15,-1,-1)]
for Threshold in Thresholds:
    print("Threshold :",Threshold)

    y_preds = np.zeros(y_test.shape)
    exec_time = np.zeros(y_test.shape)
    samples_exited = 0

    socket1.send_pyobj(np.array([Threshold]))
    trash = socket1.recv_pyobj()
    print("Sending to Edge Server The correspond Threshold")
    
    for sample in range(len(y_test)):

        input_image = np.expand_dims(x_test[sample],axis=0)
        label =y_test[sample]

        t0 = time.time()

        y_inter , y_pred = m_exit1.predict(input_image)

        entropy = compute_entropy(y_pred)
        # print("Entropy : %f" % (entropy))
        
        if entropy < Threshold:
            t1 = time.time()
            print(" ----------  COMPUTATION FINISHED AT END DEVICE ---------- ")
            print("Predicted Class : %d" % (np.argmax(y_pred,axis=-1)))
            print("Real Class : %d" % (label))
            print("EXECUTION TIME : %f s" % (t1-t0))

            y_preds[sample] = np.argmax(y_pred,axis=-1)
            exec_time[sample] = t1-t0
            # sys.exit()

        else:
            samples_exited +=1
            # print("Sending request %s …" % (sample+1))
            socket1.send_pyobj(y_inter)

            #  Get the reply.
            message = socket1.recv_pyobj()
            
            if message[0]==-1:
                # print(" ---------- WAITING FOR THE CLOUD ---------- ")
                
                message = socket2.recv_pyobj()
                
                t1 = time.time()
                print(" ----------  COMPUTATION FINISHED AT CLOUD SERVER ---------- ")
                print("Predicted Class : %d" % (message))
                print("Real Class : %d" % (label))
                print("EXECUTION TIME : %f s." % (t1-t0))

                y_preds[sample] = int(message)
                exec_time[sample] = t1-t0

            else:
                t1 = time.time()
                print(" ----------  COMPUTATION FINISHED AT EDGE SERVER ---------- ")
                print("Predicted Class : %d" % (message))
                print("Real Class : %d" % (label))
                print("EXECUTION TIME : %f s." % (t1-t0))

                y_preds[sample] = int(message)
                exec_time[sample] = t1-t0

    #%% Evaluation for ResNet-B
    accuracy = np.sum(y_preds==y_test)/len(y_test)
    exec_time = np.mean(exec_time)
    sample = [Threshold,accuracy,exec_time,samples_exited]


    with open('result.pickle','rb') as f:
        data = pickle.load(f)

    data.append(sample)
    with open("result.pickle","wb") as f:
        pickle.dump(data,f, pickle.HIGHEST_PROTOCOL)




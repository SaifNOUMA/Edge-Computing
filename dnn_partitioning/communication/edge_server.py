import time
import zmq
import sys 
from fr_utils import *
from keras.models import load_model


m_exit2 = load_model("exit2.h5")

end_device , cloud_server = sys.argv[1:]
Threshold = 0.1
# Connect to the end device
context = zmq.Context()
print("Connecting to End device…")
socket1 = context.socket(zmq.REP)
socket1.bind(end_device)   # "tcp://*:500"

# Publish to the Cloud
print("Connecting to Cloud")
socket2 = context.socket(zmq.PUB)
socket2.bind(cloud_server)   # "tcp://*:500"


nb = 1
while True :
    y_inter = socket1.recv_pyobj()

    if len(y_inter.shape)>1:
        print("Request From IoT Device N° %d" % (nb))
        y_inter , y_pred = m_exit2.predict(y_inter)
        entropy = compute_entropy(y_pred)
        if Threshold<entropy:
            #print("CONFIDENT RESULT AT EDGE SERVER \nSENDING RESULT TO END_DEVICE ...")
            label = np.argmax(y_pred,axis=-1)
            socket1.send_pyobj(np.array([label]))
        else:
            #print("NOT CONFIDENT RESULT AT EDGE SERVER \nSENDING INTERMEDIATE VECTOR TO THE CLOUD ...")
            socket1.send_pyobj(np.array([-1]))
            socket2.send_pyobj(y_inter)

        
        nb += 1
        
    else:
        Threshold = y_inter
        socket1.send_pyobj(np.array([-1]))
        print("Recieving from IoT Device the Threshold")
        nb = 0

    

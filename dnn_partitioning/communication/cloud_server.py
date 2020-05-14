import time
import zmq
import sys 

from fr_utils import *
from keras.models import load_model


m_exit3 = load_model("exit3.h5")


end_device, edge_server = sys.argv[1:]

# Publish to the End device
context = zmq.Context()
print("Connecting to End device…")
socket1 = context.socket(zmq.PUB)
socket1.bind(end_device)   # "tcp://*:500"

# Subscribe to the Edge Server
print("Connecting to Edge Server…")
socket2 = context.socket(zmq.SUB)
socket2.setsockopt(zmq.SUBSCRIBE, b"")
socket2.connect(edge_server)   # "tcp://*:500"


nb = 1
while True:
    #  Wait for next request from client
    y_inter = socket2.recv_pyobj()
    print("Received request from edge server N° %d" % nb)

    y_pred = m_exit3.predict(y_inter)
    label = np.argmax(y_pred, axis=-1)

    #  Send result back to end device
    socket1.send_pyobj(label)
    nb += 1

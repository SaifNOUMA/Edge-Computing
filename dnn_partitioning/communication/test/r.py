
import numpy as np
import zmq
import sys
import time

edge_server , cloud = sys.argv[1:]

# Publish to the edge server
context1 = zmq.Context()
print("Connecting to Edge server…")
socket1 = context1.socket(zmq.REQ)
socket1.connect(edge_server)   # "tcp://*:500"

# Subscribe to the Cloud
context2 = zmq.Context()
print("Connecting to the Cloud…")
socket2 = context2.socket(zmq.SUB)
socket2.setsockopt(zmq.SUBSCRIBE, b"")
socket2.connect(cloud)   # "tcp://*:500"
    
#  Do 10 requests, waiting each time for a response
for request in range(10):
    print("Sending request %s …" % request)
    #socket.send(b"Hello")
    socket1.send_pyobj(np.array([1,1,1]))

    #  Get the reply.
    message = socket2.recv()
    print("Received reply %s [ %s ]" % (request, message))

import numpy as np
import zmq
import sys
import time

end_device , cloud = sys.argv[1:]

# Publish to the edge server
context1 = zmq.Context()
print("Connecting to End Device…")
socket1 = context1.socket(zmq.REP)
socket1.bind(end_device)   # "tcp://*:500"

# Subscribe to the Cloud
context2 = zmq.Context()
print("Connecting to the Cloud…")
socket2 = context2.socket(zmq.PUB)
socket2.bind(cloud)   # "tcp://*:500"

while True:
    #  Wait for next request from client
    message = socket1.recv_pyobj()
    print("Received request: %s" % message)

    #  Do some 'work'
    time.sleep(1)

    #  Send reply back to client
    socket2.send_pyobj(message)
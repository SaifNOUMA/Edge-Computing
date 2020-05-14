#
#   Hello World client in Python
#   Connects REQ socket to tcp://localhost:5555
#   Sends "Hello" to server, expects "World" back
#
import numpy as np
import zmq
import sys

connect_to = sys.argv[1]
context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server…")
socket = context.socket(zmq.REQ)
socket.connect(connect_to)

#  Do 10 requests, waiting each time for a response
for request in range(10):
    print("Sending request %s …" % request)
    #socket.send(b"Hello")
    socket.send_pyobj(np.array([1,1,1]))

    #  Get the reply.
    message = socket.recv()
    print("Received reply %s [ %s ]" % (request, message))

import time
import zmq
import sys 

bind_to = sys.argv[1]
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind(bind_to)

while True:
    #  Wait for next request from client
    message = socket.recv_pyobj()
    print("Received request: %s" % message)

    #  Do some 'work'
    time.sleep(1)

    #  Send reply back to client
    socket.send(b"World")

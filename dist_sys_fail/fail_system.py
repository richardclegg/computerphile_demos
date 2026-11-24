#!/usr/bin/env python3
# node_deadlock.py
#
# A tiny "shop" made of two identical servers that talk to each other.
# It works perfectly when used slowly, and locks up forever under load.
#
# Run two copies, in two terminals:
#   python node_deadlock.py 8001 8002
#   python node_deadlock.py 8002 8001
#
# Each server has two web pages:
#   /order  - called by customers. To take an order we must ask the OTHER
#             server to check stock, so this page calls the peer's /check.
#   /check  - called by the other server. Pretends to look up a database.
#!/usr/bin/env python3
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

#We take port numbers as command line arguments
if len(sys.argv) != 3:
    print("Usage: ./fail_system.py MY_PORT PEER_PORT")
    print("Example: ./fail_system.py 8001 8002")
    sys.exit(1) 
MY_PORT = int(sys.argv[1])    # the port this server listens on
PEER_PORT = int(sys.argv[2])  # the port of the other server
 
# Only 4 requests may be worked on at once. This is our "worker pool".
# Real servers have the same kind of limit (threads, processes,
# database connections). A semaphore is a counter: taking a worker
# subtracts one, giving it back adds one, and anyone who tries to take
# a worker when the counter is zero has to wait.
workers = threading.Semaphore(4)
 
 
class Handler(BaseHTTPRequestHandler):
 
    def do_GET(self):
        # Take a worker. If all 4 are busy, wait here until one is free.
        workers.acquire()
 
        if self.path == "/order":
            # Ask the other server to check stock, and wait for its reply.
            # THE BUG: we keep hold of our worker the whole time we wait.
            url = "http://localhost:" + str(PEER_PORT) + "/check"
            urllib.request.urlopen(url).read()
 
        if self.path == "/check":
            # Pretend to do some database work.
            time.sleep(0.05)
 
        # Give the worker back so someone else can use it.
        workers.release()
 
        # Send the reply.
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok\n")
 
    def log_message(self, *args):
        # Stop the server printing a line for every request.
        pass
 
 
# Start the server. It makes a new thread for every request that arrives,
# but those threads can only do useful work once they get a worker.
print("Server on port", MY_PORT, "talking to port", PEER_PORT)
server = ThreadingHTTPServer(("", MY_PORT), Handler)
server.serve_forever()
 

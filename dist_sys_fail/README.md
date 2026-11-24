This demo is intended to create a very simple but readable example of a distributed networked system that crashes under load. The idea is two processes communicating to each other. They work fine if not loaded but crash at a higher load.

The processes act like very simple web servers returning a very basic (lazy) OK response.

To start them

./fail_system.py 8001 8002 &
./fail system.py 8002 8001 &

This will start both systems

To attack them 
./crasher.sh

If the sleep command in crasher.sh is high > 0.05 it should run count to 100 and print that 200 OK messages were received

If the sleep command is low < 0.02 it will lock the systems (you will need to kill them with kill -9) produce some messages and count that < 200 messages were received

In between maybe it will run.

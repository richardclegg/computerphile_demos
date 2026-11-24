#!/bin/sh
# load.sh - send 10 orders to each server, all at the same time.
# The & runs each curl in the background, so they all happen together.
# --max-time 5 makes curl give up after 5 seconds instead of hanging forever.

#Remove the log file to blank it.
rm -f ok.log
for i in  $(seq 1 100)
do
	curl -s --max-time 5 localhost:8001/order >> ok.log || echo "Oh no! Something bad!" &
	curl -s --max-time 5 localhost:8002/order >> ok.log || echo "Help me, I've fallen!" &
	sleep 0.05
    echo $i
done


wait    # wait for every curl to finish (or give up) -- blocks until all background tasks complete
echo "Count of the log"
wc ok.log
echo "done"


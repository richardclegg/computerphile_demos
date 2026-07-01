This is some python code for looking at the Lord of the Rings data. The data itself is from the paper "One Network to Rule them All".

You need to install python and also Raphtory. Instructions to install Raphtory are at https://raphtory.com

Once you run the code raphtory_test.py it should print a graph but also start a webserver on http://localhost:1736/

To browse the graph you need to go to that site on your web browser.
Go to "list of saved graphs" on the left hand side of the interface and select LOTR.
You can now use your mouse to zoom and pan. 

On ubuntu you probably want to use a virtual environment e.g.
python3 -m venv my-venv
my-venv/bin/pip install raphtory
my-venv/bin/pip install matplotlib
my-venv/bin/pip install pandas
my-venv/bin/python3 raphtory_test.py 




#!/usr/bin/env python

from raphtory import Graph
from raphtory import graphql
from raphtory import algorithms as alg
import raphtory as rp
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("lotr.csv")
df=df.reset_index()
lotr_graph = Graph()
lotr_graph.load_edges_from_pandas(
    df=df, time="index", src="v", dst="w"
)


pr= rp.algorithms.pagerank(lotr_graph)

top_5 = pr.top_k(50)
gandalf_rank = pr.get("Gandalf")
print(f"Gandalf's ranking is {round(gandalf_rank, 5)}\n")
frodo_rank = pr.get("Frodo")
print(f"Frodo's ranking is {round(frodo_rank, 5)}\n")
for rank, (node, score) in enumerate(top_5.items(), 1):
    print(f"Rank {rank}: {node.name} with a score of {score:.5f}")

gimportance = []
fimportance = []
time = []


for windowed_graph in lotr_graph.rolling(window=250):
    result = rp.algorithms.pagerank(windowed_graph)
    fimportance.append(result.get("Frodo"))
    gimportance.append(result.get("Gandalf"))
    
    time.append(windowed_graph.earliest_time)
    
plt.plot(time, gimportance, marker="o", label="Gandalf")
plt.plot(time, fimportance, marker="s", label="Frodo")

plt.xlabel("Index (time)")
plt.ylabel("Pagerank Score")
plt.title("Importance over time")
plt.legend()
plt.grid(True)

server = graphql.GraphServer("./tmp")
client = server.start().get_client()
client.send_graph("LOTR", lotr_graph, overwrite=True)

plt.show()

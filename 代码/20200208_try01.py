import numpy as np
from mayavi import mlab
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
data=pd.read_csv("influence_data.csv",encoding="utf-8")
'''question1画网络图，追随者指向影响者'''
origin_pop=data[(data['follower_main_genre']=='Pop/Rock')&(data['influencer_main_genre']=='Pop/Rock')][['follower_id','influencer_id','follower_active_start','influencer_active_start']]
pop=origin_pop[origin_pop['follower_active_start']<=1950]
follower=pop['follower_id'].tolist()
influencer=pop['influencer_id'].tolist()
List=[]
for i in range(len(follower)):
    List.append((follower[i],influencer[i]))
G=nx.DiGraph()
nodes=list(set(follower+influencer))
G.add_nodes_from(nodes)
G.add_edges_from(List)
H = G

# reorder nodes from 0,len(G)-1
G = nx.convert_node_labels_to_integers(H)
# 3d random layout
pos = nx.spring_layout(G, dim=3)
# numpy array of x,y,z positions in sorted node order
xyz = np.array([pos[v] for v in sorted(G)])
# scalar colors
scalars = np.array(list(G.nodes())) + 5

pts = mlab.points3d(
    xyz[:, 0],
    xyz[:, 1],
    xyz[:, 2],
    scalars,
    scale_factor=0.1,
    scale_mode="none",
    colormap="Blues",
    resolution=20,
)

pts.mlab_source.dataset.lines = np.array(list(G.edges()))
tube = mlab.pipeline.tube(pts, tube_radius=0.005)
mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8))
mlab.show()
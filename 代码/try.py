import pandas as pd
import matplotlib.pyplot as plt
final_cluster=pd.read_csv("TempData/question2_final_cluster.csv",encoding='utf-8')
data = pd.read_csv("TempData/q2_cluster_location.csv", encoding='utf-8')
final_data=pd.merge(final_cluster,data,on='id',how='inner')
final_data=final_data[['A','B','cluster']]
final_data
for i in range(final_data.shape[0]):
    x=final_data.iloc[i,0]
    y=final_data.iloc[i,1]
    cluster=final_data.iloc[i,2]
    if cluster==1:
        plt.scatter(x,y,color='Yellow',s=1)
    elif cluster==2:
        plt.scatter(x,y,color='red',s=1)
    elif cluster==3:
        plt.scatter(x,y,color='black',s=1)
    elif cluster==4:
        plt.scatter(x,y,color='blue',s=1)
    elif cluster==5:
        plt.scatter(x,y,color='purple',s=1)
plt.savefig("Picture/final_claster.png",dpi=1000)
plt.show()
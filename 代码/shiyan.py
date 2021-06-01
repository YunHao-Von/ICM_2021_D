import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
original_data=pd.read_csv("influence_data.csv",encoding="utf-8")
influencer=original_data['influencer_id'].tolist()
follower=original_data['follower_id'].tolist()
pure_follower = list(set(influencer+follower))
frame_1=original_data[['influencer_id','influencer_active_start']]
frame_2=original_data[['follower_id','follower_active_start']]
frame_1.columns = ['id','year']
frame_2.columns = ['id','year']
x=pd.concat([frame_1,frame_2],axis=0)
x.to_csv("Tempdata/temp01.csv",encoding="utf-8",index=False)
second_data=pd.read_csv("TempData/temp01.csv",encoding="utf-8")
pure_date = second_data.drop_duplicates(subset=['id', 'year'], keep='first')
result=pure_date.groupby('year')['id'].count()
result =pd.DataFrame(result)
result.to_csv("Tempdata/temp01.csv",encoding="utf-8")
plot_data=pd.read_csv("Tempdata/temp01.csv",encoding="utf-8")
year=plot_data['year'].tolist()
id=plot_data["id"].tolist()
plt.title("ALL")
plt.plot(year,id,'r-')
plt.show()
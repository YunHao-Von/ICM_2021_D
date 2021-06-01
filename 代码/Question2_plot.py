import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

original_data = pd.read_csv("influence_data.csv", encoding="utf-8")
frame_1 = original_data[['influencer_id', 'influencer_active_start', 'influencer_main_genre']]
frame_2 = original_data[['follower_id', 'follower_active_start', 'follower_main_genre']]
frame_1.columns = ['id', 'year', 'type']
frame_2.columns = ['id', 'year', 'type']
x = pd.concat([frame_1, frame_2], axis=0)
pure_information = x.drop_duplicates(subset=['id', 'year'], keep='first')
pure_information.to_csv("TempData/pure_information.csv",encoding="utf-8",index=False)
left_data=pd.read_csv("data_by_artist.csv",encoding="utf-8")
new_data=pd.merge(left_data,pure_information,left_on='artist_id',right_on='id',how='inner')
new_data=new_data.drop(columns=['id', 'year','artist_name'])
new_data.to_csv("TempData/q2new_data.csv",encoding="utf-8",index=False)

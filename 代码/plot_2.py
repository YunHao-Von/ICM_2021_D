import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

original_data = pd.read_csv("influence_data.csv", encoding="utf-8")


def get_plot_information(want_type):
    frame_1 = original_data[['influencer_id', 'influencer_active_start', 'influencer_main_genre']]
    frame_2 = original_data[['follower_id', 'follower_active_start', 'follower_main_genre']]
    frame_1.columns = ['id', 'year', 'type']
    frame_2.columns = ['id', 'year', 'type']
    x = pd.concat([frame_1, frame_2], axis=0)
    x.to_csv("Tempdata/temp01.csv", encoding="utf-8", index=False)
    second_data = pd.read_csv("TempData/temp01.csv", encoding="utf-8")
    want_type = want_type
    second_data = second_data[second_data['type'] == want_type]
    second_data = second_data[['id', 'year']]
    pure_date = second_data.drop_duplicates(subset=['id', 'year'], keep='first')
    result = pure_date.groupby('year')['id'].count()
    result = pd.DataFrame(result)
    result.to_csv("Tempdata/temp01.csv", encoding="utf-8")
    plot_data = pd.read_csv("Tempdata/temp01.csv", encoding="utf-8")
    year = plot_data['year'].tolist()
    id = plot_data["id"].tolist()
    return year, id


Share = ['Pop/Rock', 'R&B;', 'Country', 'Jazz', 'Electronic', 'Vocal', 'Reggae', 'Latin', 'Folk', 'Blues',
         'Religious', 'International', 'New Age', 'Comedy/Spoken', 'Stage & Screen', 'Classical', 'Easy Listening',
         'Avant-Garde', 'Unknown', "Children's"]
print(len(Share))
for i in range(len(Share)):
    temp_type = Share[i]
    num = i + 1
    year,id=get_plot_information(temp_type)
    plt.subplot(4,5,num)
    plt.plot(year,id,'b-')
    plt.title(temp_type)
plt.show()

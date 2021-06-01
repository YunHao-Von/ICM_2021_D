# # use feature importance for feature selection
# from numpy import loadtxt
# from numpy import sort
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.feature_selection import SelectFromModel
# import pandas as pd
# Share = {'Pop/Rock': 0,'R&B;': 1,'Country': 2,'Jazz': 3,'Electronic': 4,'Vocal': 5,'Reggae': 6,'Latin': 7,'Folk': 8,'Blues': 9,'Religious': 10,
#          'International': 11,'New Age': 12,'Comedy/Spoken': 13,'Stage & Screen': 14,'Classical': 15,'Easy Listening': 16,'Avant-Garde': 17,'Unknown': 18,"Children's": 19}
# def get_label(x):
#     x = str(x)
#     return Share[x]
# data = pd.read_csv("TempData/q2new_data.csv", encoding="utf-8")
# data["label"] = data["type"].apply(get_label)
# data = data.drop(columns=['type'])
# '''标准化'''
# def z_normal(name):
#     v_mean = data[name].mean() #均值
#     v_std = data[name].std() #标准差
#     data[name] = data[name].map(lambda x: (x - v_mean) / v_std) # z分数标准化
# z_normal('danceability');z_normal('energy');z_normal('valence');z_normal('tempo');z_normal('loudness')
# z_normal('acousticness');z_normal('instrumentalness');z_normal('liveness');z_normal('speechiness');z_normal('duration_ms');z_normal('popularity');z_normal('count')
# data.to_csv("TempData/temp06.csv",encoding="utf-8",index=False)
# y = data[['label']]
# X = data.drop(columns=['label','artist_id'])
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)
# params = {
#     'booster': 'gbtree',
#     'objective': 'multi:softmax',  # 多分类的问题
#     'num_class': 20,  # 类别数，与 multisoftmax 并用
#     'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
#     'subsample': 0.7,  # 随机采样训练样本
#     'colsample_bytree': 0.7,  # 生成树时进行的列采样
#     'min_child_weight': 3,
#     'silent': 1,  # 设置成1则没有运行信息输出，最好是设置为0.                # 如同学习率
#     'seed': 921,
#     'nthread': 4,
#     'eval_metric': 'merror',  # cpu 线程数
#     'eta': 0.007,
#     'max_depth': 10,  # 构建树的深度，越大越容易过拟合
#     'gamma': 0.1,
# }
# model = XGBClassifier(params)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# predictions = [round(value) for value in y_pred]
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
# thresholds = sort(model.feature_importances_)
# print(thresholds)
# print(sum(model.feature_importances_))
# for thresh in thresholds:
#     # select features using threshold
#     selection = SelectFromModel(model, threshold=thresh, prefit=True)
#     select_X_train = selection.transform(X_train)
#     # train model
#     selection_model = XGBClassifier()
#     selection_model.fit(select_X_train, y_train)
#     # eval model
#     select_X_test = selection.transform(X_test)
#     y_pred = selection_model.predict(select_X_test)
#     predictions = [round(value) for value in y_pred]
#     accuracy = accuracy_score(y_test, predictions)
#     print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy * 100.0))

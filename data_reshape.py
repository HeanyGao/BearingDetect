import pandas as pd
import numpy as np
train_path='D:/9课程/网课模式识别/python/train.csv'
traindata=pd.read_csv(train_path)
train_X = traindata.iloc[:,1:6001]
train_X_array = train_X.values
train_X_reshape = train_X_array.reshape(792*20,300)
train_X_ts = pd.DataFrame(train_X_reshape)
ids = np.empty([len(train_X_ts),2])
for i in range(1,len(train_X_ts)+1):
    ids[i-1][0] = i//21+1
    ids[i-1][1] = i % 21
train_X_ts['id'] = ids[:,0]
train_X_ts['time'] = ids[:,1]
train_X_ts.head()
print(train_X_ts)
train_X_ts.to_csv("feature_test.csv",index = 0,header=1)


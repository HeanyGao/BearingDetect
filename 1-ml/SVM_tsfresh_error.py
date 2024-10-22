
if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from sklearn import svm
    from sklearn.model_selection import train_test_split 
    #读取训练集和测试集
    train_path='D:/9课程/网课模式识别/python/train.csv'
    test_path='D:/9课程/网课模式识别/python/testdata.csv'
    traindata=pd.read_csv(train_path)
    testdata=pd.read_csv(test_path)
    #数据预处理
    #把属性和类别标签分开
    #train_X=traindata.drop('label',axis=1)  #drop删除某列
    #train_X.head()
    train_X = traindata.iloc[:,1:6001]

    '''
    #特征提取
    train_X_feature = pd.DataFrame()
    train_X_feature['sum'] = np.sum(train_X, axis = 1)  #在二维数组a[i][j]中，axis=1就是j从0到end,即一行 
    train_X_feature['abs_sum'] = np.sum(np.abs(train_X),axis = 1)
    train_X_feature['per5'] = np.percentile(train_X,q=5, axis = 1)  #有5%的值小于这个数，有95%的值大于这个数
    train_X_feature['per95'] = np.percentile(train_X,q=95, axis = 1)  #有5%的值小于这个数，有95%的值大于这个数
    train_X_feature['per99'] = np.percentile(train_X,q=99, axis = 1)  #有5%的值小于这个数，有95%的值大于这个数
    train_X_feature['mean'] = np.mean(train_X,axis=1)
    train_X_feature['min'] = np.min(train_X,axis=1)
    train_X_feature['max'] = np.max(train_X,axis=1)
    train_X_feature['std'] = np.std(train_X,axis=1)  #标准差
    train_X_feature['var'] = np.var(train_X,axis=1)  #方差
    train_X_feature['median'] = np.median(train_X,axis=1)  #中值
    train_X_feature['skew'] = stats.skew(train_X,axis=1)  #样本偏度，正态为0，单峰连续分布，若大于零，则意味着右边有更多权重
    train_X_feature['kurtosis'] = stats.kurtosis(train_X,axis=1) #峰度
    train_X_feature['label'] = traindata['label']
    train_X_feature.head()

    train_y = pd.DataFrame()
    train_y['label']=traindata['label']


    #训练集特征提取
    test_feature = pd.DataFrame()
    test_feature['id'] = testdata['id']
    testdata = testdata.iloc[:,1:6001]
    test_feature['sum'] = np.sum(testdata, axis = 1)  #在二维数组a[i][j]中，axis=1就是j从0到end,即一行 
    test_feature['abs_sum'] = np.sum(np.abs(testdata),axis = 1)
    test_feature['per5'] = np.percentile(testdata,q=5, axis = 1)  #有5%的值小于这个数，有95%的值大于这个数
    test_feature['per95'] = np.percentile(testdata,q=95, axis = 1)  #有5%的值小于这个数，有95%的值大于这个数
    test_feature['per99'] = np.percentile(testdata,q=99, axis = 1)  #有5%的值小于这个数，有95%的值大于这个数
    test_feature['mean'] = np.mean(testdata,axis=1)
    test_feature['min'] = np.min(testdata,axis=1)
    test_feature['max'] = np.max(testdata,axis=1)
    test_feature['std'] = np.std(testdata,axis=1)  #标准差
    test_feature['var'] = np.var(testdata,axis=1)  #方差
    test_feature['median'] = np.median(testdata,axis=1)  #中值
    test_feature['skew'] = stats.skew(testdata,axis=1)  #样本偏度，正态为0，单峰连续分布，若大于零，则意味着右边有更多权重
    test_feature['kurtosis'] = stats.kurtosis(testdata,axis=1) #峰度
    test_feature.head()

    #画图观察,在模型中没有用
    #PCA降为二维
    from sklearn.decomposition import PCA
    var=["sum","abs_sum","per5","per95","per99","mean","min","max","std","var","median","skew","kurtosis"]
    pca = PCA(n_components=2)
    pca.fit(train_X_feature[var])
    #将降维数据转换成降维后的数据，当模型训练好后，对于新输入的数据，都可以用transform方法来降维
    X_pca=pca.transform(train_X_feature[var])
    #返回 所保留的n个成分各自的方差百分比
    print(pca.explained_variance_ratio_)  
    X_pca_df =pd.DataFrame(X_pca) 
    pic_x=pd.concat([X_pca_df,train_y],axis=1)
    plt.figure(1)
    import matplotlib.pyplot as plt
    colors=['blue','orange','green','red','purple','brown','pink','gray','olive','cyan'] 
    for i in range(0,10):
        #这个画图，就是找到label = i 的那些行，第一个特征给Xi,第二个特征给yi
        Xi=pic_x[pic_x['label']==i][0]
        yi=pic_x[pic_x['label']==i][1]
        plt.scatter(Xi, yi, s=20, c=colors[i], label=i)

    plt.figure(2)
    state_list=[0,1]
    for i in state_list:
        Xi=pic_x[pic_x['label']==i][0]
        yi=pic_x[pic_x['label']==i][1]
        plt.scatter(Xi, yi, s=20, c=colors[i], label=i)
    '''
    #Tsfresh
    train_X_array = train_X.values
    train_X_reshape = train_X_array.reshape(792*2,3000)
    train_X_ts = pd.DataFrame(train_X_reshape)
    ids = np.empty([len(train_X_ts),2])
    for i in range(1,len(train_X_ts)+1):
        ids[i-1][0] = i//3+1
        ids[i-1][1] = i % 3
    train_X_ts['id'] = ids[:,0]
    train_X_ts['time'] = ids[:,1]
    train_X_ts.head()
    y = traindata['label']

    from tsfresh import extract_features, select_features
    from tsfresh.utilities.dataframe_functions import impute
    from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
    extract_settings = ComprehensiveFCParameters()
    extracted_features = extract_features(train_X_ts,column_id="id",column_sort="time",default_fc_parameters=extract_settings, impute_function=impute)
    extracted_features.head()
    # from tsfresh import select_features
    # from tsfresh.utilities.dataframe_functions import impute
    # impute(extraced_features)
    # filtered_features = select_features(extraced_features, y)
    # filtered_features.head()

    #划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(train_X_feature[var], train_y, random_state=1, train_size=0.8)
    print("Size of training set:{} size of testing set:{}".format(x_train.shape[0],x_test.shape[0]))

    #训练SVM模型
    clf = svm.SVC(C=1, kernel='rbf', gamma=0.01, decision_function_shape='ovr')
    #训练
    clf.fit(x_train, y_train)


    from sklearn.metrics import classification_report, confusion_matrix
    #在测试集上的分类结果
    y_hat = clf.predict(x_test)
    print(confusion_matrix(y_test,y_hat))
    print(classification_report(y_test,y_hat))



    #测试数据
    clf.fit(train_X_feature[var],train_y)
    y_hat2=clf.predict(test_feature[var])
    print(y_hat2)
    test_feature['label']=y_hat2
    test_feature.to_csv("Result for SVM.csv",index = 0,header=1,columns=['id','label'])
    #dataframe=pd.DataFrame({'id':test_feature['id'],'label':y_hat2})
    #dataframe.to_csv("Result_SVM.csv",index=False,sep=',')
    plt.show()
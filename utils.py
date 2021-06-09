import numpy as np
from sklearn.preprocessing import MinMaxScaler 
    
def single_minmaxscale(data, scale_range):
    def minmaxscale(data, scale_range):
        scaler = MinMaxScaler(scale_range)
        scaler.fit(data)
        normalized = scaler.transform(data)
        return normalized

    X = []
    for i in data:
        X.append(minmaxscale(i.reshape(-1,1), scale_range))
    return np.asarray(X)     


def data_preproc(dataset, scale_range = (-1, 1)):
    X_tra, y_tra, X_tst, y_tst = dataset
    X_tra = single_minmaxscale(X_tra, scale_range)
    X_tst = single_minmaxscale(X_tst, scale_range)

    X_tra = X_tra.astype('float32')
    X_tra = X_tra.reshape(-1,1,120,1)
    X_tst = X_tst.astype('float32')
    X_tst = X_tst.reshape(-1,1,120,1)
    print('Finished preprocessing.')
    return (X_tra, y_tra, X_tst, y_tst)



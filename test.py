from torch import cuda
import numpy as np


# device= cuda.is_available()

def hamming_score(y_true, y_pred, normalize= True, sample_wight= None):
    acc_list= []
    for i in range(y_true.shape[0]):
        set_true= set(np.where(y_true[i])[0])
        set_pred= set(np.where(y_pred[i])[0])
        tmp_a= None
        if len(set_true) == 0 and len(set_pred)== 0:
            tmp_a= 1
        else:
            tmp_a= len(set_true.intersection(set_pred))/\
                float(len(set_true.union(set_pred)))

        acc_list.append(tmp_a)
    return np.mean(acc_list)

y_pred= np.array([1,2,10,4])
y_true= np.array([1,2,10,4])

loss= hamming_score(y_true, y_pred)
print(f"The hamming loss is {loss}")
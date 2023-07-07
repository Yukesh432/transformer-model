from sklearn.metrics import hamming_loss
import numpy as np

'''In multiclass classification, the Hamming loss corresponds to the Hamming distance between y_true and
y_pred which is equivalent to the subset zero_one_loss function, when normalize parameter is set to True.

In multilabel classification, the Hamming loss is different from the subset zero-one loss. The zero-one loss
considers the entire set of labels for a given sample incorrect if it does not entirely match the true set of labels. 
Hamming loss is more forgiving in that it penalizes only the individual labels.

Reference: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html
'''



y_pred= [1,2,10,4]
y_true= [2,2,32,4]

loss= hamming_loss(y_true, y_pred)
print(f"The hamming loss is {loss}")

x= np.where(y_pred[1])[0]
print(x)
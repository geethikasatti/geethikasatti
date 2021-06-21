
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter

x_train = pd.read_csv('Diabetes_XTrain.csv')
y_train = pd.read_csv('Diabetes_YTrain.csv')

X = x_train.values
Y = y_train.values




def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))
def knn(X,Y,querypoint,k=17):
    vals = []
    m = X.shape[0]
    for i in range(m):
        d = dist(querypoint,X[i])
        vals.append((d,Y[i]))
    
    vals = sorted(vals)
    vals = vals[:k]
    vals = np.array(vals)
    #print(vals)
    new_vals = np.unique(vals[:,1],return_counts = True)
    index = new_vals[1].argmax()
    pred = new_vals[0][index]
    return pred
    

X1 = pd.read_csv('Diabetes_Xtest.csv')
X_test = X1.values

Y1 = pd.read_csv('sample_submission.csv')
Y_test = Y1.values


number = X_test.shape[0]
pred = []
for i in range(number):
    
    n = knn(X,Y,X_test[i])
    pred.append(n)

#print(pred)
p = ",".join([str(elem) for elem in pred])
p = p.replace(",","\n")
p = p.replace("[","")
p = p.replace("]","")
    
with open("pred.csv","w") as file:
   file.write("Outcome\n"+p)

print(np.mean(pred==Y_test))


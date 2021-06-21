# Using KNN Algorithm to predict if a person will have diabetes or not
# importing libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


# loading the dataset

data = pd.read_csv('diabetes.csv')

print(data.head())

# Replace columns like [Gluscose,BloodPressure,SkinThickness,BMI,Insulin] with Zero as values with mean of respective column


zero_not_accepted = ['Glucose','BloodPressure','SkinThickness','BMI','Insulin']
# for col in zero_not_accepted:
#     for i in data[col]:
#         if i==0:
#             colSum = sum(data[col])
#             meanCol=colSum/len(data[col])
#             data[col]=meanCol

for col in zero_not_accepted:
    data[col]= data[col].replace(0,np.NaN)
    mean = int(data[col].mean(skipna=True))
    data[col] = data[col].replace(np.NaN,mean)

plt.figure(1)
plt.subplot(121),sns.distplot(data['Glucose'])
plt.subplot(122),data['Glucose'].plot.box(figsize=(16,5))
plt.show()  
plt.figure(2)
plt.subplot(121),sns.distplot(data['BloodPressure'])
plt.subplot(122),data['BloodPressure'].plot.box(figsize=(16,5))
plt.show() 
plt.figure(3)
plt.subplot(121),sns.distplot(data['SkinThickness'])
plt.subplot(122),data['SkinThickness'].plot.box(figsize=(16,5))
plt.show() 
    

# extracting independent variables

X = data.iloc[:,0:8]
# extracting dependent variable

y = data.iloc[:,8]
# Explorning data to know relation before processing

sns.heatmap(data.corr())

plt.figure(figsize=(25,7))
sns.countplot(x='Age',hue='Outcome',data=data,palette='Set1')

# splitting dataset into training and testing set


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# feature scaling

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# loading model - KNN

classifier = KNeighborsClassifier(n_neighbors=11,p=2,metric='euclidean')
# fitting model

classifier.fit(X_train,y_train)

# making predictions

y_pred = classifier.predict(X_test)
# evaluating model

conf_matrix = confusion_matrix(y_test,y_pred)
print(conf_matrix)
print(f1_score(y_test,y_pred))

# accuracy

print(accuracy_score(y_test,y_pred))

plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


warnings.filterwarnings('ignore')


# In[3]:


df=pd.read_excel('diabetes.xlsx')


# In[4]:


df.shape


# In[5]:


df.head()


# In[14]:


#import pandas_profiling as pp


# In[15]:


#pp.ProfileReport(df)   


# In[16]:


df.describe()


# In[17]:


df['Outcome'].value_counts()


# In[18]:


x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[21]:


from sklearn.linear_model import LogisticRegression


# In[22]:


model=LogisticRegression()


# In[23]:


model.fit(x_train,y_train)


# In[24]:


y_pred=model.predict(x_test)


# In[25]:


from sklearn import metrics


# In[26]:


cm=metrics.confusion_matrix(y_test,y_pred)
pd.DataFrame(cm,index=['NO','YES'],columns=['NO','YES'])


# In[27]:
confusion_matrix = confusion_matrix(y_test,y_pred)
print(confusion_matrix)
print(f1_score(y_test,y_pred))    


print(metrics.accuracy_score(y_test,y_pred))
# Compare Algorithms
import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# prepare configuration for cross validation test harness
seed = 3
# prepare models
models = []
models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
#!/usr/bin/env python
# coding: utf-8

# # # BHARAT INTERN
# # NAME-LINGAMPALLI JASWANTH
# # TASK1-STOCK PREDICTION
# - IN THIS WE WILL USE THE stock-market-prediction-and-forecasting-using-stac
# 
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# In[2]:


df = pd.read_csv('../input/nsetataglobalbeverageslimited/NSE-Tata-Global-Beverages-Limited.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


df1 = df.reset_index()['Close']


# In[10]:


df1


# In[11]:


plt.plot(df1)
plt.title('Stacked index view')


# In[12]:


from sklearn.preprocessing import MinMaxScaler
scaler =  MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[13]:


df1


# In[14]:


training_size=int(len(df1)*0.65)        # split data train & test
test_size=len(df1)-training_size                                   
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[15]:


training_size


# In[16]:


test_size


# In[17]:


train_data[0:5]


# In[18]:


# convert an array of values into a Dataset matrix
import numpy
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, dataX:0,1,2,3-----99   dataY:100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)


# In[19]:


time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[20]:


print(X_train.shape), print(y_train.shape)


# In[21]:


print(X_test.shape), print(ytest.shape)


# In[22]:


# Reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[23]:


get_ipython().system('pip install tensorflow')


# In[24]:


### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[25]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[26]:


model.summary()


# In[27]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)


# In[28]:


#Lets Do the prediction and check Performance Metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[29]:


#Transform back to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[30]:


# Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[31]:


#Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))


# In[32]:


#Plotting Data
# shift train predictions for plotting
look_back=100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.figure(figsize=(15,6))
plt.plot(scaler.inverse_transform(df1), '-b', label='Train Data')
plt.plot(trainPredictPlot,'--r', label='Train Predict',linewidth=2.0)
plt.plot(testPredictPlot,'g', label='Test Predict', linewidth=2.0)
leg = plt.legend();
plt.show()


# In[ ]:





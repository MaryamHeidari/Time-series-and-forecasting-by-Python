#!/usr/bin/env python
# coding: utf-8

# In[123]:


from dateutil.parser import parse 
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import numpy as np
import pandas as pd


# In[39]:


df_batch = pd.read_csv("Desktop/Northeastern University/term5, winter 2020/capston data set/batch_scores.csv")
df_employee = pd.read_csv("Desktop/Northeastern University/term5, winter 2020/capston data set/employee_profile.csv")
df_allocator = pd.read_csv("Desktop/Northeastern University/term5, winter 2020/capston data set/allocator.csv")
df_ideal = pd.read_csv("Desktop/Northeastern University/term5, winter 2020/capston data set/ideal_times.csv")
df_preprocessed = pd.read_csv("Desktop/Northeastern University/term5, winter 2020/capston data set/preprocessed_data.csv")
df_quanta = pd.read_csv("Desktop/Northeastern University/term5, winter 2020/capston data set/quanta_map.csv")
df_wheels = pd.read_csv("Desktop/Northeastern University/term5, winter 2020/capston data set/wheels.csv")


# In[40]:


df_batch.head(5)


# In[41]:


df_batch.describe()


# In[42]:


# checking missinh values in batch score dataset
print(df_batch.isnull().sum())


# In[43]:


df_employee.head(5)


# In[44]:


#Removing missinh values of employee profile dataset 
df_employee = df_employee.dropna(axis=0, subset=['Shift','Supervisor ID'])


# In[45]:


# checking missinh values in employee profile dataset
#Leaving the missing values of the email variable intact as we are not working with email
print(df_employee.isnull().sum())


# In[46]:


df_allocator.head(5)


# In[47]:


#Spliting the Trigger variable
# new data frame with split value columns 
new1 = df_allocator["TRIGGER"].str.split("|", n = 1, expand = True) 
  
# making separate TRIGGER type column from new data frame 
df_allocator["TRIGGER_type"]= new1[0] 
  
# making separate Department column from new data frame 
df_allocator["Department"]= new1[1] 
  
# Dropping Divisions/Department columns 
df_allocator.drop(columns =["TRIGGER"], inplace = True) 
  
# df display 
df_allocator.head(10) 


# In[48]:


# checking missinh values in allocator dataset
print(df_allocator.isnull().sum())


# In[49]:


df_allocator.describe()


# In[50]:


df_ideal.head(5)


# In[51]:


# checking missinh values in ideal times dataset
print(df_ideal.isnull().sum())


# In[52]:


df_ideal.describe()


# In[53]:


df_preprocessed.head(5)


# In[54]:


#Spliting the Trigger variable
# new data frame with split value columns 
new2 = df_preprocessed["TRIGGER"].str.split("|", n = 1, expand = True) 
  
# making separate TRIGGER type column from new data frame 
df_preprocessed["TRIGGER_type"]= new2[0] 
  
# making separate Department column from new data frame 
df_preprocessed["Department"]= new2[1] 
  
# Dropping Divisions/Department columns 
df_preprocessed.drop(columns =["TRIGGER"], inplace = True) 
  
# df display 
df_preprocessed.head(10) 


# In[55]:


# checking missinh values in preprocessed data dataset
print(df_preprocessed.isnull().sum())


# In[56]:


df_preprocessed.describe()


# In[57]:


df_quanta.head(5)


# In[58]:


# checking missinh values in quanta map dataset
print(df_quanta.isnull().sum())


# In[59]:


df_quanta.describe()


# In[60]:


#imputing the NULL values with 0
df_wheels = df_wheels.fillna(0)
df_wheels.head(10)


# In[61]:


# checking missinh values in wheels dataset
print(df_wheels.isnull().sum())


# In[62]:


df_wheels.describe()


# In[63]:


#combining employee profile and batch score datasets
df4 = pd.merge(df_employee, df_batch, left_on='Employee_ID', right_on='EMPLOYEE', how='inner',indicator=True)
df4.head(10)


# In[64]:


#Removing variables that are not required for our analysis
df_combine= df4.drop(['EMAIL','Shift','Systems','EMPLOYEE','_merge'],axis=1)
df_combine.head(10)


# In[65]:


# new data frame with split value columns 
new = df_combine["Divisions/Department"].str.split("/", n = 1, expand = True) 
  
# making separate Divisions column from new data frame 
df_combine["Divisions"]= new[0] 
  
# making separate Department column from new data frame 
df_combine["Department"]= new[1] 
  
# Dropping Divisions/Department columns 
df_combine.drop(columns =["Divisions/Department"], inplace = True) 
  
# df display 
df_combine.head(10) 


# In[66]:


#Maryam's Part


# In[171]:


# Draw Plot
def plot_df(df, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y)
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()


# In[172]:


Employee_ID = np.unique(df_combine.Employee_ID)
Employee_ID


# In[200]:


#individual for each employee ID=RXWP49G
individual1= df_combine.loc[df_combine['Employee_ID'] == 'RXWP49G']
df1= individual1[['MASTER','CONSISTENCY','QUALITY_IMPACT','CAPACITY_UTILIZATION','ADHERENCE','VOLUME','DATE']]
df1


# In[128]:


import matplotlib.pyplot as plt
df1.plot(x= 'DATE', y=['MASTER','CONSISTENCY','QUALITY_IMPACT','CAPACITY_UTILIZATION','ADHERENCE','VOLUME'])
df1.plot(x= 'DATE', y='MASTER',color='blue')
df1.plot(x= 'DATE', y='CONSISTENCY', color='orange')
df1.plot(x= 'DATE', y='QUALITY_IMPACT', color= 'green')
df1.plot(x= 'DATE', y='CAPACITY_UTILIZATION',color='red')
df1.plot(x= 'DATE', y='ADHERENCE', color='violet')
df1.plot(x= 'DATE', y='VOLUME', color='brown')

plt.show()


# In[165]:


#group, Average
Divisions = np.unique(df_combine.Divisions)
for i in np.arange(len(Divisions)):
    df_temp= df_combine.loc[df_combine['Divisions'] == Divisions[i]]
    df_temp1=np.unique(df_temp['DATE'])
    column_names= ["DATE","AVG_MASTER","AVG_CONSISTENCY","AVG_QUALITY_IMPACT","AVG_CAPACITY_UTILIZATION","AVG_ADHERENCE","AVG_VOLUME"]
    DT=pd.DataFrame(index=np.arange(len(df_temp1)),columns = column_names)
    for j in np.arange(len(df_temp1)):
        DT.iloc[j]['DATE']= df_temp1[j]
        dt1= df_temp.loc[df_temp['DATE'] == df_temp1[j]]
        DT.iloc[j]['AVG_MASTER']= np.mean(dt1.MASTER)
        DT.iloc[j]['AVG_CONSISTENCY']= np.mean(dt1.CONSISTENCY)
        DT.iloc[j]['AVG_QUALITY_IMPACT']= np.mean(dt1.QUALITY_IMPACT)
        DT.iloc[j]['AVG_CAPACITY_UTILIZATION']= np.mean(dt1.CAPACITY_UTILIZATION)
        DT.iloc[j]['AVG_ADHERENCE']= np.mean(dt1.ADHERENCE)
        DT.iloc[j]['AVG_VOLUME']= np.mean(dt1.VOLUME)

    
    DT.plot(x= 'DATE', y=['AVG_MASTER','AVG_CONSISTENCY','AVG_QUALITY_IMPACT','AVG_CAPACITY_UTILIZATION','AVG_ADHERENCE','AVG_VOLUME'], title=Divisions[i])  
    plt.show()


# In[99]:


#group, STD
Divisions = np.unique(df_combine.Divisions)
for i in np.arange(len(Divisions)):
    df_temp= df_combine.loc[df_combine['Divisions'] == Divisions[i]]
    df_temp1=np.unique(df_temp['DATE'])
    column_names= ["DATE","STD_MASTER","STD_CONSISTENCY","STD_QUALITY_IMPACT","STD_CAPACITY_UTILIZATION","STD_ADHERENCE","STD_VOLUME"]
    DT=pd.DataFrame(index=np.arange(len(df_temp1)),columns = column_names)
    for j in np.arange(len(df_temp1)):
        DT.iloc[j]['DATE']= df_temp1[j]
        dt1= df_temp.loc[df_temp['DATE'] == df_temp1[j]]
        DT.iloc[j]['STD_MASTER']= np.std(dt1.MASTER)
        DT.iloc[j]['STD_CONSISTENCY']= np.std(dt1.CONSISTENCY)
        DT.iloc[j]['STD_QUALITY_IMPACT']= np.std(dt1.QUALITY_IMPACT)
        DT.iloc[j]['STD_CAPACITY_UTILIZATION']= np.std(dt1.CAPACITY_UTILIZATION)
        DT.iloc[j]['STD_ADHERENCE']= np.std(dt1.ADHERENCE)
        DT.iloc[j]['STD_VOLUME']= np.std(dt1.VOLUME)
    
    DT.plot(x= 'DATE', y=['STD_MASTER','STD_CONSISTENCY','STD_QUALITY_IMPACT','STD_CAPACITY_UTILIZATION','STD_ADHERENCE','STD_VOLUME'], title=Divisions[i])  
    plt.show()


# In[ ]:


# forecasting


# In[179]:


conda install -c conda-forge fbprophet 


# In[259]:


pip install pmdarima


# In[229]:


from statsmodels.tsa.stattools import adfuller
from numpy import log
result = adfuller(df1.MASTER.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])


# In[304]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df1.MASTER); axes[0, 0].set_title('Original Series')
plot_acf(df1.MASTER, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(df1.MASTER.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(df1.MASTER.diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(df1.MASTER.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df1.MASTER.diff().diff().dropna(), ax=axes[2, 1])


plt.show()


# In[249]:


#AR
# PACF plot of 1st differenced series
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})
fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df1.MASTER.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(df1.MASTER.diff().dropna(), ax=axes[1])

plt.show()


# In[250]:


#MA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df1.MASTER.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,1.2))
plot_acf(df1.MASTER.diff().dropna(), ax=axes[1])

plt.show()


# In[305]:


#ARIMA Model
from statsmodels.tsa.arima_model import ARIMA

# 1,1,2 ARIMA Model
model = ARIMA(df1.MASTER, order=(1,1,2))
model_fit = model.fit(disp=0)
print(model_fit.summary())


# In[306]:


# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='MASTER', ax=ax[1])
plt.show()


# In[307]:


# Actual vs Fitted
model_fit.plot_predict(dynamic=False)
plt.show()


# In[257]:


from statsmodels.tsa.stattools import acf

# Create Training and Test
train = df1.MASTER[:416]
test = df1.MASTER[416:]

# Build Model  
model = ARIMA(train, order=(1,1,2))  
fitted = model.fit(disp=-1)  

# Forecast
fc, se, conf = fitted.forecast(104, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# In[258]:


# Build Model
model = ARIMA(train, order=(3, 2, 1))  
fitted = model.fit(disp=-1)  
print(fitted.summary())

# Forecast
fc, se, conf = fitted.forecast(104, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# In[260]:


# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})

forecast_accuracy(fc, test.values)


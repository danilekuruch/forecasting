import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters

#File reading
file_path_string = filedialog.askopenfilename()
df = pd.read_csv(file_path_string, usecols=['NA_Sales','EU_Sales','JP_Sales','Other_Sales'])
print(df)
arr=df.values
n=0 #time series number: 0 -'NA_Sales',1 - 'EU_Sales', 2 - 'JP_Sales', 3 -'Other_Sales'

#Plot an initial time series
fig = plt.figure()
plt.plot(arr[:200,n], color='red', linewidth=2);
plt.show()

register_matplotlib_converters()
sns.set_style("darkgrid")
plt.rc("figure", figsize=(16, 12))
plt.rc("font", size=13)

#Time series component plotting function
def add_stl_plot(fig, res, legend):
    """Add 3 plots from a second STL fit"""
    axs = fig.get_axes()
    comps = ["trend", "seasonal", "resid"]
    for ax, comp in zip(axs[1:], comps):
        series = getattr(res, comp)
        if comp == "resid":
            ax.plot(series, marker="o", linestyle="none")
        else:
            ax.plot(series)
            if comp == "trend":
                ax.legend(legend, frameon=False)
                
#Defining MAPE function
def MAPE(Y_actual,Y_Predicted):
    numerator =np.zeros((len(Y_actual)))
    for u in range(len(Y_actual)):
        if Y_actual[u]==0:
           numerator[u] = (Y_Predicted[u]-Y_actual[u])
        else:
           numerator[u] = (Y_Predicted[u]-Y_actual[u])/Y_actual[u]
    mape = np.median(np.abs(numerator))*100
    return mape

#Defining U2 function
def u2(Y_actual,Y_Predicted):
    Y_Predicted_1 =Y_Predicted[1:]
    Y_actual_1=Y_actual[1:]
   
    numerator =np.zeros((len(Y_actual)))
    for u in range(len(Y_actual)-1):
        if Y_actual[u]==0:
           numerator[u] = (Y_Predicted[u+1]-Y_actual[u+1])
        else:
            numerator[u] = (Y_Predicted[u+1]-Y_actual[u+1])/Y_actual[u]
    denominator =np.zeros((len(Y_actual)))
    for u in range(len(Y_actual)-1):
        if Y_actual[u]==0:
           denominator[u] = (Y_actual[u+1]-Y_actual[u])
        else:
            denominator[u] = (Y_actual[u+1]-Y_actual[u])/Y_actual[u]
    
    return np.sqrt(np.sum(np.square(numerator))/np.sum(np.square(denominator)))

#Defining U1 function
def u1(Y_actual,Y_Predicted):
    
    numerator =np.zeros((len(Y_actual)))
    for u in range(len(Y_actual)):
        numerator[u] = (Y_Predicted[u]-Y_actual[u])
    numerator = np.sqrt(np.median(np.square(numerator)))
      
    return numerator/(np.sqrt(np.median(np.square(Y_Predicted)))+np.sqrt(np.median(np.square(Y_actual))))

#Testing stationarity
from statsmodels.tsa.stattools import adfuller
print(adfuller(arr[:300,n]))

#Testing normality
import scipy.stats as stats
print(stats.jarque_bera(arr[:300,n]))

#Moving average and std plotting function
def get_stationarity(timeseries, window):
    
    # rolling statistics
    rolling_mean=np.zeros((300,1))
    rolling_std=np.zeros((300,1))
    for u in range(300-window+1):
        rolling_mean[u] = timeseries[u:(u+window-1)][:].mean()
        rolling_std[u] = timeseries[u:(u+window-1)][:].std()
    
    # rolling statistics plot
    fig = plt.figure()
    original = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Mean')
    std = plt.plot(rolling_std, color='black', label='Std')
    plt.legend(loc='best')
    plt.title('Mean & Standard Deviation')
    plt.show(block=False)
    
#Moving average and std plotting
get_stationarity(arr[:300,n],12)#12- window size

#STL decomposition (robust, nonrobust)                                                                       
stl = STL(np.log(arr[:200,n]+1), period=3, robust=True)
res_robust = stl.fit()
fig = res_robust.plot()
res_non_robust = STL(np.log(arr[:200,n]+1), period=3, robust=False).fit()
add_stl_plot(fig, res_non_robust, ["Robust", "Non-robust"])
plt.show()

print("Non-robust trend MAPE",MAPE(np.log(arr[:200,n]+1),res_non_robust.trend))
print("Non-robust trend U1", u1(np.log(arr[:200,n]+1),res_non_robust.trend))
print("Non-robust trend U2", u2(np.log(arr[:200,n]+1),res_non_robust.trend))

print("Non-robust trend and seasonality MAPE",MAPE(np.log(arr[:200,n]+1),res_non_robust.trend+res_non_robust.seasonal))
print("Non-robust trend and seasonality U1", u1(np.log(arr[:200,n]+1),res_non_robust.trend+res_non_robust.seasonal))
print("Non-robust trend and seasonality U2", u2(np.log(arr[:200,n]+1),res_non_robust.trend+res_non_robust.seasonal))

print("Robust trend MAPE",MAPE(np.log(arr[:200,n]+1),res_robust.trend))
print("Robust trend U1", u1(np.log(arr[:200,n]+1),res_robust.trend))
print("Robust trend U2", u2(np.log(arr[:200,n]+1),res_robust.trend))

print("Robust trend and seasonality MAPE",MAPE(np.log(arr[:200,n]+1),res_robust.trend+res_robust.seasonal))
print("Robust trend and seasonality U1", u1(np.log(arr[:200,n]+1),res_robust.trend+res_robust.seasonal))
print("Robust trend and seasonality U2", u2(np.log(arr[:200,n]+1),res_robust.trend+res_robust.seasonal))

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast

#STL forecasting nonrobust
stlf = STLForecast(np.log(arr[:200,n]+1), ARIMA, model_kwargs=dict(order=(2, 1, 0)), period=5, robust=False)
stlf_res = stlf.fit()

T=6
forecast = stlf_res.forecast(T)
plt.plot(np.log(arr[:200+T,n]+1))
plt.plot(range(200,200+T),forecast,color='red')
plt.show()

print("Forecasting nonrobust MAPE", MAPE(np.log(arr[200:200+T,n]+1),forecast))
print("Forecasting nonrobust U1",u1(np.log(arr[200:200+T,n]+1),forecast))
print("Forecasting nonrobust U2",u2(np.log(arr[200:200+T,n]+1),forecast))

#STL forecasting robust
stlf = STLForecast(np.log(arr[:200,n]+1), ARIMA, model_kwargs=dict(order=(2, 1, 0)), period=5, robust=True)
stlf_res = stlf.fit()

forecast = stlf_res.forecast(T)
plt.plot(np.log(arr[:200+T,n]+1))
plt.plot(range(200,200+T),forecast,color='red')
plt.show()

print("Forecasting robust MAPE", MAPE(np.log(arr[200:200+T,n]+1),forecast))
print("Forecasting robust U1",u1(np.log(arr[200:200+T,n]+1),forecast))
print("Forecasting robust U2",u2(np.log(arr[200:200+T,n]+1),forecast))
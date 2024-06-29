import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.graphics.tsaplots import plot_predict
import numpy as np
import datetime
import yfinance as yf
import statsmodels.api as sm
import seaborn as sns
from arch import arch_model
import arch.univariate as univariatearch
from scipy.stats import chi2
from scipy.stats import t
from scipy.stats import norm
from scipy.optimize import fmin, minimize
from sklearn.metrics import mean_absolute_error
import plotly.express as px
import plotly.figure_factory as ff
from arch import arch_model
from ipywidgets import HBox, VBox, Dropdown, Output
from math import inf
from IPython.display import display
import bs4 as bs
import requests
import yfinance as yf
import plotly.io as pio
pio.renderers.default='browser'
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

os.chdir("D:\EBS Universität\Semester 3 ST 2023\Empirical Finance\Presentation\code")

# h is the cond. std deviation based on Garch, r is the simulated return
def calc_h1(time, table):
    return 0.01+0.05*(table.iloc[time-1, 4]**2) + 0.94*table.iloc[time-1, 2]

def calc_h2(time, table):
    return 0.5+0.2*(table.iloc[time-1, 5]**2) + 0.5*table.iloc[time-1, 3]

def calc_r1(time, table):
    return np.sqrt(table.iloc[time, 2]) * table.iloc[time, 0]

def calc_r2(time, table):
    return np.sqrt(table.iloc[time, 3]) * table.iloc[time, 1]

#initialize the data table using different episilon
def initdata(epsilon):
    data = pd.DataFrame()
    data["e1"] = epsilon[: ,0]
    data["e2"] = epsilon[: ,1]
    data[["h1","h2","r1","r2"]] = np.zeros_like(len(data), shape=(1000, 4))
    # Init values
    data.iloc[0, 2:6] = (0.01, 0.5, np.sqrt(0.01)*data.iloc[0, 0], np.sqrt(0.5)*data.iloc[0, 1])
    return data

def calculate_table(table):
    for t in range(1, len(table)):
        # calculate h1
        table.iloc[t, 2] = calc_h1(t, table)
        # calculate h2
        table.iloc[t, 3] = calc_h2(t, table)
        # calculate r1
        table.iloc[t, 4] = calc_r1(t, table)
        # calculate r2
        table.iloc[t, 5] = calc_r2(t, table)
    return table
 
def moving_average_correlation_f2(ts1, ts2, window_size):
    """
    Calculates the moving average correlation between two time series using a rolling window of a specified size.
    
    Args:
        ts1 (pandas.Series): the first time series
        ts2 (pandas.Series): the second time series
        window_size (int): the size of the rolling window used to compute the correlation
        
    Returns:
        pandas.Series: a series containing the moving average correlation values
    """
    # Concatenate the two time series into a DataFrame
    df = pd.concat([ts1, ts2], axis=1)
    # Compute the rolling correlation using a window of the specified size
    rolling_corr = df.rolling(window_size).corr().iloc[::2,-1]
    # Compute the rolling mean of the correlation values
    ma_corr = rolling_corr.groupby(level=0).mean()
    return ma_corr


def moving_average_correlation(ts1, ts2, window_size):
    """
    Calculates the moving average correlation between two time series using a rolling window of a specified size.
    
    Args:
        ts1 (pandas.Series): the first time series
        ts2 (pandas.Series): the second time series
        window_size (int): the size of the rolling window used to compute the correlation
        
    Returns:
        pandas.Series: a series containing the moving average correlation values
    """
    # Concatenate the two time series into a DataFrame
    df = pd.concat([ts1, ts2], axis=1)
    # Compute the rolling correlation using a window of the specified size
    rolling_corr = df.rolling(window_size).corr().iloc[::2,-1]
    return rolling_corr

def E06_correlation(ts1, ts2, lmbda):
    """
    Calculates the EWMA with 0.94 smoothing factor correlation between two time series
    
    Args:
        ts1 (pandas.Series): the first time series
        ts2 (pandas.Series): the second time series
        lmbda (int) : smoothing factor
        
    Returns:
        pandas.Series: a series containing the moving average correlation values
    """
    y = pd.concat([ts1, ts2], axis=1).values
    T = len(y)
    EWMA = np.full([T,3], np.nan)
    lmbda = 0.94
    S = np.cov(y, rowvar = False)
    EWMA[0,] = S.flatten()[[0,3,1]]
    for i in range(1,T):
        S = lmbda * S + (1-lmbda) * np.transpose(np.asmatrix(y[i-1]))* np.asmatrix(y[i-1])
        EWMA[i,] = [S[0,0], S[1,1], S[0,1]]
    EWMArho = np.divide(EWMA[:,2], np.sqrt(np.multiply(EWMA[:,0],EWMA[:,1])))
    return EWMArho

### Implementing DCC

def vector_l(matrix):
    lower_matrix = np.tril(matrix, k=-1)
    array_with_zero = np.matrix(lower_matrix).A1
    array_without_zero = array_with_zero[array_with_zero != 0]
    return array_without_zero


def dcc_function_to_max(theta, udata):
    N, T = np.shape(udata)
    llf = np.zeros((T, 1))
    trdata = np.array(norm.ppf(udata).T, ndmin=2)
    Rt, vector_lRt = matrix_operations(theta, trdata)
    for i in range(0, T):
        llf[i] = -0.5 * np.log(np.linalg.det(Rt[:, :, i]))
        llf[i] = llf[i] - 0.5 * np.matmul(np.matmul(trdata[i, :], (np.linalg.inv(Rt[:, :, i]) - np.eye(N))),
                                          trdata[i, :].T)
    llf = np.sum(llf)

    return -llf


def matrix_operations(theta, trdata):
    T, N = np.shape(trdata)

    a, b = theta

    if min(a, b) < 0 or max(a, b) > 1 or a + b > .999999:
        a = .9999 - b

    Qt = np.zeros((N, N, T))

    Qt[:, :, 0] = np.cov(trdata.T)

    Rt = np.zeros((N, N, T))
    vector_lRt = np.zeros((T, int(N * (N - 1) / 2)))

    Rt[:, :, 0] = np.corrcoef(trdata.T)

    for j in range(1, T):
        Qt[:, :, j] = Qt[:, :, 0] * (1 - a - b)
        Qt[:, :, j] = Qt[:, :, j] + a * np.matmul(trdata[[j - 1]].T, trdata[[j - 1]])
        Qt[:, :, j] = Qt[:, :, j] + b * Qt[:, :, j - 1]
        Rt[:, :, j] = np.divide(Qt[:, :, j], np.matmul(np.sqrt(np.array(np.diag(Qt[:, :, j]), ndmin=2)).T,
                                                       np.sqrt(np.array(np.diag(Qt[:, :, j]), ndmin=2))))

    for j in range(0, T):
        vector_lRt[j, :] = vector_l(Rt[:, :, j].T)
    return Rt, vector_lRt

model_parameters = {}
udata_list = []


def run_garch_on_return(rets, udata_list, model_parameters):
    for x in rets:
        am = arch_model(rets[x], dist='t')
        short_name = x.split()[0]
        model_parameters[short_name] = am.fit(disp='off')

        # udata = garch_t_to_u(rets[x], model_parameters[short_name])
        # udata = t.cdf((rets[x] - res.params['mu']) / res.conditional_volatility, res.params['nu'])
        
        udata = t.cdf(
            (rets[x] - model_parameters[short_name].params['mu']) / model_parameters[short_name].conditional_volatility,
            model_parameters[short_name].params['nu'])

        udata_list.append(udata)
    return udata_list, model_parameters

def run_dcc(ts1, ts2, disp=False):
    """
    Calculates the in-sample dcc correlation for return series 1 and return series 2
    
    Args:
        ts1 (pandas.Series): the first time series
        ts2 (pandas.Series): the second time series
        
    Returns:
        pandas.Series: a series containing predicted in-sample correlation values
    """
    rets = pd.concat([ts1, ts2], axis=1)
    model_parameters = {}
    udata_list = []
    udata_list, model_parameters = run_garch_on_return(rets.dropna(), udata_list, model_parameters)
    constraints = ({'type': 'ineq', 'fun': lambda x: -x[0] - x[1] + 1})
    bounds = ((0, 0.5), (0, 0.9997))

    opt_out = minimize(dcc_function_to_max, [0.01, 0.95], args=(udata_list,), bounds=bounds, constraints=constraints)


    llf = dcc_function_to_max(opt_out.x, udata_list)
    if(disp == True):
        print("opt_out.success: ", opt_out.success)
        print("opt_out.x", opt_out.x)
        print("llf: ", llf)


    trdata = np.array(norm.ppf(udata_list).T, ndmin=2)
    Rt, vector_lRt = matrix_operations(opt_out.x, trdata)
    stock_names = [x.split()[0] for x in rets.iloc[:, :5].columns]
    corr_name_list = []
    for i, name_a in enumerate(stock_names):
        if i == 0:
            pass
        else:
            for name_b in stock_names[:i]:
                corr_name_list.append(name_a + "-" + name_b)

    dcc_corr = pd.DataFrame(vector_lRt, index=rets.iloc[:, :5].dropna().index, columns=corr_name_list)
    
    return dcc_corr

############################################            main

raw_data = pd.read_csv('data_5.csv', index_col=0)
asset_1_name = raw_data.columns[0]
asset_2_name = raw_data.columns[1]
asset_1 = raw_data[asset_1_name]
asset_2 = raw_data[asset_2_name]
# data_series_length = 

rets = ((raw_data / raw_data.shift(1)) - 1).dropna(how='all') * 100

udata_list, model_parameters = run_garch_on_return(rets.dropna(), udata_list, model_parameters)

# rets.tail()

constraints = ({'type': 'ineq', 'fun': lambda x: -x[0] - x[1] + 1})
bounds = ((0, 0.5), (0, 0.9997))

opt_out = minimize(dcc_function_to_max, [0.01, 0.95], args=(udata_list,), bounds=bounds, constraints=constraints)

print("opt_out.success: ", opt_out.success)
print("opt_out.x", opt_out.x)


llf = dcc_function_to_max(opt_out.x, udata_list)
print("llf: ", llf)


trdata = np.array(norm.ppf(udata_list).T, ndmin=2)
Rt, vector_lRt = matrix_operations(opt_out.x, trdata)
stock_names = [x.split()[0] for x in rets.iloc[:, :5].columns]
corr_name_list = []
for i, name_a in enumerate(stock_names):
    if i == 0:
        pass
    else:
        for name_b in stock_names[:i]:
            corr_name_list.append(name_a + "-" + name_b)




#####################################   generate multivariate epilison

rho1 = 0.9
rho1_list = np.full(1000, 0.9)
epsilon = np.random.multivariate_normal(np.array([0,0]), np.array([[1, rho1],[rho1, 1]]), size=1000)
print(epsilon)

data1 = initdata(epsilon)
# start calculation for further h and r
data1 = calculate_table(data1)
plt.plot(data1.index, data1["r1"])
plt.show()

numlist = np.arange(0, 1000, dtype=int)
phisine = 0.5 + 0.4*np.cos(2*np.pi*numlist/200)
plt.plot(numlist, phisine)
phivec =  [np.array([[1, p], [p, 1]]) for p in phisine]
epsilon = np.zeros(shape=(1000,2))
for index, vec in enumerate(phivec):
    # print(np.random.multivariate_normal(np.array([0,0]), vec, size=1)[0])
    epsilon[index] = np.random.multivariate_normal(np.array([0,0]), vec, size=1)[0]

data2 = initdata(epsilon)
# start calculation for further h and r
data2 = calculate_table(data2)
    
plt.plot(data2.index, data2["r1"])
plt.show()

fastsine = 0.5 + 0.4*np.cos(2*np.pi*numlist/20)
plt.plot(numlist, fastsine)
phivec =  [np.array([[1, p], [p, 1]]) for p in fastsine]
epsilon = np.zeros(shape=(1000,2))
for index, vec in enumerate(phivec):
    # print(np.random.multivariate_normal(np.array([0,0]), vec, size=1)[0])
    epsilon[index] = np.random.multivariate_normal(np.array([0,0]), vec, size=1)[0]

data3 = initdata(epsilon)
# start calculation for further h and r
data3 = calculate_table(data3)
    
plt.plot(data3.index, data3["r1"])
plt.show()

step = 0.9 - 0.5*(numlist > 500)
plt.plot(numlist, step)
phivec =  [np.array([[1, p], [p, 1]]) for p in step]
epsilon = np.zeros(shape=(1000,2))
for index, vec in enumerate(phivec):
    # print(np.random.multivariate_normal(np.array([0,0]), vec, size=1)[0])
    epsilon[index] = np.random.multivariate_normal(np.array([0,0]), vec, size=1)[0]

data4 = initdata(epsilon)
# start calculation for further h and r
data4 = calculate_table(data4)
    
plt.plot(data4.index, data4["r1"])
plt.show()

ramp = (numlist/200) % 1
plt.plot(numlist, ramp)
phivec =  [np.array([[1, p], [p, 1]]) for p in ramp]
epsilon = np.zeros(shape=(1000,2))
for index, vec in enumerate(phivec):
    # print(np.random.multivariate_normal(np.array([0,0]), vec, size=1)[0])
    epsilon[index] = np.random.multivariate_normal(np.array([0,0]), vec, size=1)[0]

data5 = initdata(epsilon)
# start calculation for further h and r
data5 = calculate_table(data5)
    
plt.plot(data5.index, data5["r1"])
plt.show()


#####################################           Combine all return data

data_all = {
    "data1" : data1,
    "data2" : data2,
    "data3" : data3,
    "data4" : data4,
    "data5" : data5
}

ma100_table = pd.DataFrame()
for data in data_all:
    ma100_table[data] = moving_average_correlation(data_all[data]["r1"], data_all[data]["r2"], 100)

print(ma100_table.tail())

# plt.plot(range(len(ma100)), ma100)
# plt.show()

ewma_table = pd.DataFrame()
for data in data_all:
    ewma_table[data] = E06_correlation(data_all[data]["r1"], data_all[data]["r2"], 0.94)

print(ewma_table.tail())


dcc_table = pd.DataFrame()
for data in data_all:
    dcc_table[data] = run_dcc(data_all[data]["r1"], data_all[data]["r2"])
    
print(dcc_table.tail())


# combine all corresponding predicted data and rho data
rho_all = {
    "data1" : rho1_list,
    "data2" : phisine,
    "data3" : fastsine,
    "data4" : step,
    "data5" : ramp
}

mae_results = pd.DataFrame()
for data in data_all:
    mae_results[data] = [mean_absolute_error(ma100_table[data].dropna(), rho_all[data][0:901]),
                         mean_absolute_error(ewma_table[data].dropna(), rho_all[data]),
                         mean_absolute_error(dcc_table[data].dropna(), rho_all[data])]
    
mae_results.index = ["MA100", "EX .06", "DCC"]
print(mae_results.transpose())




######################################################## Plots

dcc_corr = pd.DataFrame(vector_lRt, index=rets.iloc[:, :5].dropna().index, columns=corr_name_list)
dcc_plot = px.line(dcc_corr, title='Dynamic Conditional Correlation plot', width=1001, height=500)

dcc_plot.show()
garch_vol_df = pd.concat(
    [pd.DataFrame(model_parameters[x].conditional_volatility / 100) * 1600 for x in model_parameters], axis=1)
garch_vol_df.columns = stock_names
px.line(garch_vol_df, title='GARCH Conditional Volatility', width=1001, height=500).show()
px.scatter(garch_vol_df, x=asset_1_name, y=asset_2_name, width=1001, height=500, title='GARCH Volatility').show()
px.line(np.log((1 + rets.iloc[:, :5].dropna() / 100).cumprod()), title='Cumulative Returns', width=1000,
        height=500).show()
rets.loc[:, [asset_1_name, asset_2_name]].corr()


output_graphics = Output()
pair_dropdown = Dropdown(options=[''] + corr_name_list)

### update_corr_data

# moving avg corr plots
ma_corr = moving_average_correlation_f2(asset_1, asset_2, window_size=50)
print(ma_corr)
px.line(ma_corr)
###

# pair_dropdown.observe(update_corr_data, 'value')

a1corr = rets.corr().values[0][1]
a1dcc = pd.DataFrame(vector_lRt[:, 0], index=rets.index)
a1dcc.columns = ['DCC']
a1dcc['corr'] = a1corr
a1dcc['ma_corr'] = ma_corr
corr_line_plot = px.line(a1dcc, title='DCC vs unconditional correlation vs Moving Average' , width=1001,
                         height=500)
# output_graphics.clear_output()

with output_graphics:
    display(corr_line_plot)

## adding stock prices to corr plot

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Scatter(x=a1dcc.index, y=a1dcc['DCC'], name="DCC", mode='lines+markers'),
    secondary_y=False,)

fig.add_trace(
    go.Scatter(x=a1dcc.index, y=a1dcc['corr'], name="Corr", mode='lines+markers'),
    secondary_y=False,)

fig.add_trace(
    go.Scatter(x=a1dcc.index, y=a1dcc['ma_corr'], name="Moving Avg Corr", mode='lines+markers'),
    secondary_y=False,)

fig.add_trace(
    go.Scatter(x=a1dcc.index, y=asset_1, name=asset_1_name, mode='lines'),
    secondary_y=True,)

fig.add_trace(
    go.Scatter(x=a1dcc.index, y=asset_2, name=asset_2_name, mode='lines'),
    secondary_y=True,)

with output_graphics:
    display(fig)


##########


VBox([pair_dropdown, output_graphics])

print("MODEL PARAMETERS:")
print(model_parameters)
print("exit")





# pppp1 = px.line(asset_1)

# pppp1.add_trace(px.line(asset_2))

# ### overlap figures
# fig = go.Figure()

# fig.add_trace(
#     go.Scatter(x=asset_1.index,
#                y=asset_1
#     ))

# fig.add_trace(
#     go.Bar(x=asset_2.index,
#                y=asset_2
#     ))

# fig.show()

# plt.plot(ma_corr.index, ma_corr)
# plt.show()
#
# plt.plot(asset_1.index, asset_1)
# plt.plot(asset_2.index, asset_2)
# plt.show()

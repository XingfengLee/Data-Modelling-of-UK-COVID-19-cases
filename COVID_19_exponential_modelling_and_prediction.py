import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt

# data from the following website:
dStr  = 'https://www.arcgis.com/apps/opsdashboard/index.html#/f94c3c90da5b4e9f9a0b19484dd4bb14/'

data  = np.array([2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 8, 8, 8, 9, 9,
	9, 9, 9, 9, 9, 9, 9, 9, 9,  13, 13, 13, 13, 19, 23, 35, 40,
	51, 85, 114, 160, 206, 271, 321, 373, 456, 595, 797, 1061, 
	1391, 1543, 1950, 2626, 3269,3269+714,3269+714+1035,5683,6650,
	 8077, 9529, 11658,14543,17089,19522,22141,25150,29474,33718,
	 38168,41903,47806,51608,55242,60733,65077,70272,73758,78991,84279]) 
# 3269 is number of cases on 19th March 2020
# 3269+714+1035 is the total number of postive case on 21st Mar 2020.
# 5683 is the total number of postive case on 22nd Mar 2020.
# 6650, = 5683+ 967, on 23rd Mar, 2020, Staying at home and away from others (social distancing) policy begin today ; 
# 19522, on 29th Mar, 2020
# 29474 on 1st April, 2020

  
 
today = dt.date.today()
date  = today.strftime("%b-%d-%Y")
 


N     = data.shape[0]
oldData = data[0:49]

# nWeek = 1 # number of week you want to predict from now
# nDay  = N + nWeek*7 + 1 

n     = 3 # number of days to predict
nDay  = N + 1 + n 
time  = np.arange(1,N+1)
oldTime = time[0:49]


def exp_func(x, a, b):
	# This the the exponential function to model covid-19
    return a * np.exp(b * x) # y = a*exp(b*x) 

# params, params_covariance = optimize.curve_fit(exp_func, 
# 									time, data,p0 = [0.001, 1] )
params, params_covariance = optimize.curve_fit(exp_func, 
									time, data,p0 = [0.032, 0.2361] )



time = np.arange(1,nDay)
y    = exp_func(time, params[0], params[1])
y2   = np.zeros(y.shape)

now  = dt.date(2020, 1, 31) #31 Jan 2020
then = now + dt.timedelta(days=nDay-1)
days = mdates.drange(now,then,dt.timedelta(days=1))

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3))
plt.plot(days,y, 'bs-', label='line 1', linewidth=2)

y2[0:N] = data
y2[N :nDay] = None
plt.plot(days,y2, 'ro-', label='line 1', linewidth=2)
plt.ylabel("Population (Y)")
plt.grid()

plt.title('UK COVID-19 Number of Confirmed Positive Cases Prediction on ' + date)  
plt.gcf().autofmt_xdate()

death_rate = 0.044
y3 = exp_func(time[nDay-2], params[0], params[1])
z = y3*death_rate
print(y2 )

plt.text(days[nDay-2],y[nDay-2]+4000, '  @ 4.4% ' )
plt.text(days[nDay-12],y[nDay-2]+4000, 'Number of death: ' + str(int(np.round(z)))) 
model = str( "{0:.3f}".format(params[0])) + '*exp(' + str("{0:.3f}".format(params[1]))+'*t)'
plt.text(days[1],y[nDay-5], 'Latest Prediction model is: Y = ' + model, fontsize=15)
plt.text(days[1],y[nDay-2]+50000,'Data from GOV web:  ' + dStr, fontsize=10)

# Plot the old model produced on 20th March
# old model prediction

oParams, oParams_covariance = optimize.curve_fit(exp_func, 
									oldTime, oldData,p0 = [0.032, 0.2361] )

y4 = exp_func(time, oParams[0], oParams[1])
plt.plot(days, y4, 'y*--', label='line 1', linewidth=1)



# Staying at home and away from others (social distancing) on 23/03/2020, data[52]=23/03/2020
sData   = data[0:53]
sTime   = time[0:53]
sParams, sParams_covariance = optimize.curve_fit(exp_func, 
									sTime, sData,p0 = [0.032, 0.2361] )
y5 = exp_func(time, sParams[0], sParams[1])
plt.plot(days, y5, 'g+--', label='line 1', linewidth=1)


plt.legend(('Latest Prediction', 'Original Data', 'Prediction on 20/03/2020', 'Model on 23/03/2020 (GOV Stay at Home Policy)'))
plt.show()



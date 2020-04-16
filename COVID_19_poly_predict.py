import numpy as np 
import matplotlib.dates as mdates
import datetime as dt
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly

# data from the following website:
dStr  = 'https://coronavirus.data.gov.uk'

data  = np.array([2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 8, 8, 8, 9, 9,
	9, 9, 9, 9, 9, 9, 9, 9, 9,  13, 13, 13, 13, 19, 23, 35, 40,
	51, 85, 114, 160, 206, 271, 321, 373, 456, 595, 797, 1061, 
	1391, 1543, 1950, 2626, 3269,3983,5018,5683,6650,8077,
	9529, 11658,14543,17089,19522,22141,25150,29474,33718,
	38168,41903,47806,51608,55242,60733,65077,70272,73758,78991,
	84279,88621,93873,98476]) # data of coronavirus cases

p     = 23
n     = 4  # number of days to predict minus one

N     = data.shape[0] # 
nDay  = N + n 

x     = np.arange(1,N+1)
coef  = np.polyfit(x, data, p)
print(coef)
pre   = np.poly1d(coef)
yFit  = pre(x)

x_new = np.arange(1,nDay)
pcoef = coef[::-1]
fFit  = poly.Polynomial(pcoef)    
yPred = fFit(x_new)

def R2_and_F(yPred,yTrue,p,N):

	yM    = np.mean(yTrue) 
	yMean = np.full(yTrue.shape, yM)
	yTmp  = yTrue-yMean;
	sst   = np.dot(yTmp.T,yTmp)
	yPtmp = yTrue - yPred;
	sse   = np.dot(yPtmp.T,yPtmp)
	R2    = 1 - sse/sst
	F = ((sst-sse)/p) / (sse/(N-p-1))
	return R2 , F 

R2,F = R2_and_F(data,yFit,p,N)

today = dt.date.today()
date  = today.strftime("%b-%d-%Y")
now   = dt.date(2020, 1, 31) #31 Jan 2020
then  = now + dt.timedelta(days=nDay-1)
days  = mdates.drange(now,then,dt.timedelta(days=1))

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3))

plt.plot(days,yPred, 'bs-', label='line 1', linewidth=2)

y2          = np.zeros(x_new.shape)
y2[0:N]     = data
y2[N :nDay] = None
plt.plot(days,y2, 'ro-', label='line 1', linewidth=2)
plt.ylabel("Population (Y)")
plt.grid()

plt.title('UK COVID-19 Number of Confirmed Positive Cases Prediction on ' + date)  
plt.gcf().autofmt_xdate()

plt.text(days[1],yFit[nDay-20-n] , 'Goodness-of-fit: R^2 = ' + str(int(np.round(R2,3)))) 
fStr = 'F-Statistic: F('   + str(p) + ', ' + str(N-p-1) + ') =  '  +  str(int(np.round(F,3)) )
plt.text(days[1],yFit[nDay-25-n] ,  fStr) 

plt.text(days[1],yFit[nDay-20]+40000,'Data from GOV web:  ' + dStr, fontsize=10)

plt.legend(('Prediction', 'Original Data'))
plt.show()


# imports
from pandas import read_csv
import pandas as pd
import datetime
from datetime import date
from matplotlib import pyplot
from matplotlib.dates import DateFormatter
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

# break dates into years
def parser(x):
    
    return date(int(x),1,1).year

# read through the visualization csv file and get the desired past values to predict from
series = read_csv(r'C:\Users\prran\Extracurriculars\MathModeling\Modeling-the-future\lyme-data-historic.csv',index_col=0,parse_dates=[0],date_parser=parser,header=0)
series = series.astype(float)
selected_series = series.iloc[:, 1]
X = selected_series.values


# assign(0.66x) of the past values to train the ARIMA model
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]


predictions = list()

years = series.index
train_years = years[:size]
test_years = series.index[size:len(X)]

# create and train the ARIMA model
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,1))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    year = test_years[t]

# check root mean square error to determine whether the model is a good fit
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)


last_year = test_years[-1].year
# forecast until 2030

forecast_years = pd.date_range(start=str(2020), end='2030', freq='Y')

# generate predictions for the range of forecast_years
for year in forecast_years:
    model = ARIMA(history, order=(5,1,1))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    history.append(yhat)    
  #  print('predicted=%f' % (yhat),'year: ',year)

extended_years = list(test_years) + list(forecast_years)


_ey_export = []
for y in extended_years:
    if y.year != 1970:
        _ey_export.append(y.year)
    
# export the predictions into a csv file 
csv_export_df = pd.DataFrame( _ey_export,predictions[-len(_ey_export):])
scope = "testing"
csv_export_df.to_csv(f'{scope}_lime_disease_prediction_forecast.csv' )


pretty_graph_years = []

for yr in years:
    pretty_graph_years.append(yr.year)
pretty_graph_years


# plotting the predictions
pyplot.plot(_ey_export, predictions[-len(_ey_export):], color='black', label='Predicted (Future)')
pyplot.xlabel('Year')
pyplot.ylabel('Value')
pyplot.legend()
pyplot.title('ARIMA Model: Predicting Future CO2 Emissions for _ Energy')
pyplot.show()


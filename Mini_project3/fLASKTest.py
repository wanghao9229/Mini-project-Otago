import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt
from io import BytesIO
from flask import Flask, render_template, send_file, make_response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

#%matplotlib inline

plt.rcParams['figure.figsize'] = (20, 10)
plt.style.use('ggplot')

data_f = pd.read_csv('C:/Users/Andrew/Desktop/weekly sales and labour cost for all shops 2013 to 20177.csv')
data = pd.read_csv('C:/Users/Andrew/Desktop/weekly sales and labour cost for all shops 2013 to 20177.csv',
                   index_col='start_date', parse_dates=True)
# shopID = input("Enter your shop id")
shopID1 = 17
# if shopID1<min(data.shop_id) or shopID1>max(data.shop_id):
# print("Enter correct shop id number")
# return select_model()

data2 = data[
    ['sales_id', 'shop_id', 'week_no', 'sales_amount', 'item_sold', 'transactions', 'total_tax', 'sales_status']]
df1 = data2[data2.shop_id == shopID1]  # input №1
df2 = df1[df1.sales_status != 0]
df2.week_no.isnull().values.any()
nulldetect = df1.week_no.isnull()
nulldetect[nulldetect == True].index
df2.week_no.loc[nulldetect == True] = 54
df2['week_no'] = df2.week_no - 2

dff = df2[['sales_amount']]
data3 = dff.reset_index()
data4 = data3

data5 = data4.rename(columns={'start_date': 'ds', 'sales_amount': 'y'})
data5.set_index('ds')
# y.plot()
data5['y'] = np.log(data5['y'])
data5.set_index('ds')
model = Prophet()
model.fit(data5)
future = model.make_future_dataframe(periods=52, freq='w')
forecast = model.predict(future)
data5.set_index('ds', inplace=True)
forecast.set_index('ds', inplace=True)
viz_df = dff.join(forecast[['yhat', 'yhat_lower', 'yhat_upper']], how='outer')
viz_df['yhat_rescaled'] = np.exp(viz_df['yhat'])
dff.index = pd.to_datetime(dff.index)  # make sure our index as a datetime object
connect_date = dff.index[-2]  # select the 2nd to last date
mask = (forecast.index > connect_date)
predict_df = forecast.loc[mask]
viz_df = dff.join(predict_df[['yhat', 'yhat_lower', 'yhat_upper']], how='outer')
viz_df['yhat_scaled'] = np.exp(viz_df['yhat'])
ii = len(dff.sales_amount) - 1
viz_df.yhat_scaled[ii:]
predicted_future_sales = pd.DataFrame(viz_df.yhat_scaled[ii:])
predicted_future_sales1 = predicted_future_sales.rename(columns={'yhat_scaled': 'future_sales'})
predicted_future_sales2 = predicted_future_sales1.reset_index()
week_no = predicted_future_sales2['index'].dt.week
future_sales = predicted_future_sales2['future_sales']
future_sales1 = round(future_sales, 2)
start_date = predicted_future_sales2['index']
predict_data = {'future_sales': future_sales1, 'week_no': week_no, 'start_date': start_date}
predict_data1 = pd.DataFrame(predict_data)

p2 = predict_data1.set_index('start_date')
r = []
for jj in pd.DataFrame(df2.index.year.values).drop_duplicates().index.values:
    sale_year = df2.sales_amount[str(int(pd.DataFrame(df2.index.year).drop_duplicates().loc[jj]))].mean()
    r.append(sale_year)
years = pd.DataFrame(df2.index.year).drop_duplicates().start_date.values
holday = []
for t in years[0:len(years) - 1]:
    h = df2.sales_amount[df2.week_no >= 50][str(t)].mean() + df2.sales_amount[df2.week_no <= 3][str(int(t) + 1)].mean()
    holday.append(h)
year_last = p2.future_sales[p2.week_no >= 50][str(years[-1])].mean() + p2.future_sales[p2.week_no <= 3].mean()  # 2018
holday.append(year_last)
N = len(r)
Holiday_Means = holday
All_Year_Means = r
ind = np.arange(N)  # the x locations for the groups
# the width of the bars: can also be len(x) sequence



weekNO = 24

app = Flask(__name__)


@app.route('/fig/')
def fig():
    # print("Predicted sales amount for shop #"+str(shopID)+": ",int(predict_data1.future_sales[predict_data1.week_no==weekNO].values))
    # diff=predict_data1.future_sales[0]-df2.sales_amount[-1]
    # print("Difference between last actual sale and forecasted :+",diff)
    fig, ax1 = plt.subplots()
    ax1.plot(viz_df.sales_amount)
    ax1.plot(viz_df.yhat_scaled, color='black', linestyle=':')
    ax1.fill_between(viz_df.index, np.exp(viz_df['yhat_upper']), np.exp(viz_df['yhat_lower']), alpha=0.5,
                     color='darkgray')
    ax1.set_title('Sales (Orange) vs Sales Forecast (Black) for shop ' + str(shopID1))
    ax1.set_ylabel('Dollar Sales')
    ax1.set_xlabel('Dates')

    L = ax1.legend()  # get the legend
    L.get_texts()[0].set_text('Actual Sales')  # change the legend text for 1st plot
    L.get_texts()[1].set_text('Forecasted Sales')  # change the legend text for 2nd plot
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    # plt.show()
    return send_file(img, mimetype='image/png')

@app.route('/fig1/')
def fig1():
    fig1 = plt.subplots()
    bar1 = plt.bar(ind,Holiday_Means,label='Holidays')
    bar2 = plt.bar(ind,All_Year_Means,label='Avg sales per year')

    plt.ylabel('Sales_amount')
    plt.title('Holiday sales (Xmas & NY) vs Average sales per year (shop #%s)'%shopID1)
    plt.xticks(ind, (pd.DataFrame(df2.index.year).drop_duplicates().start_date.values))
    plt.legend()
    canvas = FigureCanvas(fig1)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')
# plt.show()
if __name__ == '__main__':
    app.run()

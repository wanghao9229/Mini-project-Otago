from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from numpy import inf
from scipy import signal
from fbprophet import Prophet
import matplotlib.pyplot as plt
from sklearn import linear_model
import mpld3
import os
import sqlite3
from io import BytesIO
import io
from flask import Flask, request, session, g, redirect, url_for, abort,render_template, flash
from flask_cors import *
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config.from_object(__name__) # load config from this file , flaskr.py

@app.route('/')
def show_entries():
    return render_template('main.html')

@app.route('/input/')
def my_form():
    return render_template('input.html')

@app.route('/input/', methods=['POST'])
def my_form_post():
    shop_Id = request.form['shop_id']
    week_No = request.form['week_no']
    ShopID = int(shop_Id)
    WeekNo=int(week_No)


    # %matplotlib inline

    #plt.rcParams['figure.figsize'] = (20, 10)
    plt.style.use('ggplot')

    data_f = pd.read_csv('C:/weekly sales and labour cost for all shops 2013 to 20177.csv')
    data = pd.read_csv('C:/weekly sales and labour cost for all shops 2013 to 20177.csv',
                       index_col='start_date', parse_dates=True)
    # shopID = input("Enter your shop id")
    shopID1 = ShopID
    # if shopID1<min(data.shop_id) or shopID1>max(data.shop_id):
    # print("Enter correct shop id number")
    # return select_model()
    WeekNo1=WeekNo

    data2 = data[['sales_id', 'shop_id', 'week_no', 'sales_amount', 'item_sold', 'transactions', 'total_tax', 'sales_status']]
    df1 = data2[data2.shop_id == shopID1]  # input №1
    df2 = df1[df1.sales_status != 0]
    df2.week_no.isnull().values.any()
    nulldetect = df1.week_no.isnull()
    nulldetect[nulldetect == True].index
    df2.week_no.loc[nulldetect == True] = 54
    df2['week_no'] = df2.week_no - 2
    if len(df2.week_no) > 51:

        dff = df2[['sales_amount']]
        data3 = dff.reset_index()
        data4 = data3

        data5 = data4.rename(columns={'start_date': 'ds', 'sales_amount': 'y'})
        data5.set_index('ds')
        data5 = data5.replace([np.inf, -np.inf], np.nan).fillna(0)
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

        predict_data2 = predict_data1.set_index('start_date')
        frames = [df2.sales_amount, predict_data2.future_sales]
        join = pd.concat(frames)
        detrend_sdata = signal.detrend(join)
        trend = join - detrend_sdata

        p2 = predict_data1.set_index('start_date')
        r = []
        for jj in pd.DataFrame(df2.index.year.values).drop_duplicates().index.values:
            sale_year = df2.sales_amount[str(int(pd.DataFrame(df2.index.year).drop_duplicates().loc[jj]))].mean()
            r.append(sale_year)
        years = pd.DataFrame(df2.index.year).drop_duplicates().start_date.values
        holday = []
        for t in years[0:len(years) - 1]:
            h = df2.sales_amount[df2.week_no >= 50][str(t)].mean() + df2.sales_amount[df2.week_no <= 3][str(int(t) + 1)].mean()
            holday.append(h / 2)
        year_last = p2.future_sales[p2.week_no >= 50][str(years[-1])].mean() + p2.future_sales[p2.week_no <= 3].mean()  # 2018
        holday.append(year_last / 2)
        N = len(r)
        Holiday_Means = holday
        All_Year_Means = r
        ind = np.arange(N)

        avg_sale=round(df2.sales_amount.mean(),2)
        maxSale=round(max(df2.sales_amount), 2)
        minSale=round(min(df2.sales_amount), 2)
        itemTrans=round((df2.item_sold / df2.transactions).mean(), 2)


        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(viz_df.sales_amount)
        ax1.plot(viz_df.yhat_scaled,color='green')
        ax1.plot(join.index, trend, color='blue', alpha=0.5, label='Trend')
        #ax1.plot(join.index, trend, color='blue', alpha=0.5, label='Trend')
        #ax1.fill_between(viz_df.index, np.exp(viz_df['yhat_upper']), np.exp(viz_df['yhat_lower']), alpha=0.5,
                         #color='darkgray')
        ax1.set_title('Sales (Orange) vs Sales Forecast (Green) for shop ' + str(shopID1))
        ax1.set_ylabel('Sales amount')
        ax1.set_xlabel('Dates')

        L = ax1.legend()  # get the legend
        L.get_texts()[0].set_text('Actual Sales')  # change the legend text for 1st plot
        L.get_texts()[1].set_text('Forecasted Sales')  # change the legend text for 2nd plot
        graph = mpld3.fig_to_html(fig)

        fig, ax2 = plt.subplots(figsize=(7, 4))
        bar_width = 0.4
        opacity = 0.8
        bar1 = ax2.bar(ind, Holiday_Means, bar_width, opacity, label='Holidays')
        bar2 = ax2.bar(ind + bar_width, All_Year_Means, bar_width, opacity, label='Avg sales per year')
        ticks = pd.DataFrame(df2.index.year).drop_duplicates().start_date.values

        ax2.set_ylabel('Sales_amount')
        ax2.set_title('Holiday sales (Xmas & NY) vs Average sales per year (shop #%s)' % shopID1)
        plt.xticks(ind+0.25,ticks)
        ax2.legend()
        graph1 = mpld3.fig_to_html(fig)



        f_sale=str(float(predict_data1.future_sales[predict_data1.week_no == WeekNo1].values[0]))
        n_week = WeekNo1
        id_shop = shopID1
        sale_mean=avg_sale
        max_sale=maxSale
        min_sale=minSale
        item_trans=itemTrans

    else:

        a = df2[['sales_amount', 'shop_id', 'week_no', 'transactions', 'item_sold']]
        y = a.iloc[:, 0]
        x = a.iloc[:, 3:5]
        # print (df2)
        from sklearn import linear_model
        regr2 = linear_model.LinearRegression()
        X1 = x
        y1 = y
        regr2.fit(X1, y1)
        y_predictions = regr2.predict(X1)
        y_predictions1 = pd.DataFrame(y_predictions)
        d = {'actual sales': y, 'predicted sales': y_predictions1}
        d1 = np.array(d)

        dates = pd.date_range(y.index[-1], periods=52, freq='W-MON', format='%Y-%m-%d')
        dates1 = pd.DataFrame(dates)

        mean_week_item = []
        for i in dates.week:
            mean_item_sold = a.item_sold[a.week_no == i].mean()
            mean_week_item.append(mean_item_sold)
        mean_week_item1 = pd.DataFrame(mean_week_item)

        trans_week_item = []
        for i1 in dates.week:
            mean_trans_sold = a.transactions[a.week_no == i1].mean()
            trans_week_item.append(mean_trans_sold)

        sales_week = []
        for ii1 in dates.week:
            mean_sales_sold = a.sales_amount[a.week_no == ii1].mean()
            sales_week.append(mean_sales_sold)

        dd = {'date': dates, 'weeks_no': dates.week, 'sales': sales_week, 'mean_item': mean_week_item,
              'mean_trans': trans_week_item}
        dd1 = pd.DataFrame(dd)

        dff1 = data_f[data_f.sales_status != 0]
        nulldetect = dff1.week_no.isnull()
        dff1.week_no.loc[nulldetect == True] = 54
        dff1['week_no'] = dff1.week_no - 2
        X_Cluster = dff1[['shop_id', 'sales_amount']]
        from sklearn.cluster import KMeans
        kmeans_model = KMeans(n_clusters=3, random_state=8).fit(X_Cluster)
        y_hat = kmeans_model.labels_  # clusters
        cen = kmeans_model.cluster_centers_
        y_hat1 = pd.DataFrame(y_hat)
        group_low_sales = X_Cluster[y_hat == 0]
        group_middle_sales = X_Cluster[y_hat == 2]
        group_high_sales = X_Cluster[y_hat == 1]

        fff = []
        for j in X_Cluster.shop_id:
            dfdf = X_Cluster.sales_amount[X_Cluster.shop_id == j].mean()
            fff.append(dfdf)
        f3 = pd.DataFrame(X_Cluster.shop_id.drop_duplicates())
        f4 = pd.DataFrame(fff)
        f5 = f4.drop_duplicates()
        f3['salle'] = f5.values

        # from sklearn.cluster import KMeans
        Xx2 = f3[['shop_id', 'salle']]
        kmeans_model2 = KMeans(n_clusters=3, random_state=8).fit(Xx2)
        y_hat2 = kmeans_model2.labels_  # clusters
        cen2 = kmeans_model2.cluster_centers_

        group_middle_sales2 = Xx2[y_hat2 == 0]
        group_high_sales2 = Xx2[y_hat2 == 2]
        group_low_sales2 = Xx2[y_hat2 == 1]

        nullweeks = dd1.weeks_no[dd1.mean_trans.isnull() == True]

        if (group_low_sales2.shop_id.values == shopID1).any() == True:
            cx = int(group_low_sales.sales_amount[group_low_sales.shop_id == shopID1].values.mean())
            trt = group_low_sales[group_low_sales.sales_amount > cx - 3000]
            trt2 = trt[trt.sales_amount < cx + 3000]
            valid_cls = dff1[['sales_amount', 'item_sold', 'transactions', 'week_no']].loc[trt2.index.values]
            #print("Cluster of shop %s is low sales" % shopID1)
        elif (group_middle_sales2.shop_id.values == shopID1).any() == True:
            # valid_cls=dff1[['sales_amount','item_sold','transactions','week_no']].loc[group_middle_sales.shop_id.index.values]
            cx = int(group_middle_sales.sales_amount[group_middle_sales.shop_id == shopID1].values.mean())
            trt = group_middle_sales[group_middle_sales.sales_amount > cx - 3000]
            trt2 = trt[trt.sales_amount < cx + 3000]
            valid_cls = dff1[['sales_amount', 'item_sold', 'transactions', 'week_no']].loc[trt2.index.values]
            #print("Cluster of shop %s is average sales" % shopID1)
        elif (group_high_sales2.shop_id.values == shopID1).any() == True:
            # valid_cls=dff1[['sales_amount','item_sold','transactions','week_no']].loc[group_high_sales.shop_id.index.values]
            cx = int(group_high_sales.sales_amount[group_high_sales.shop_id == shopID1].values.mean())
            trt = group_high_sales[group_high_sales.sales_amount > cx - 4000]
            trt2 = trt[trt.sales_amount < cx + 4000]
            valid_cls = dff1[['sales_amount', 'item_sold', 'transactions', 'week_no']].loc[trt2.index.values]
            #print("Cluster of shop %s is high sales" % shopID1)
        drr = valid_cls
        #print('Avg sales per week for whole period ',
        avg_sale=round(df2.sales_amount.mean(), 2)  # avg sales per week for whole period
        # avg_items_week=round(df2.item_sold[df2.week_no==17].mean(),2)# avg items for input week
        #print('Avg items sold per week for whole period ',
        #round(df2.item_sold.mean(), 2) # avg items per week for whole period
        # avg_trans_week=round(df2.transactions[df2.week_no==17].mean(),2)# avg trans for input week
        #print('Avg trans per week for whole period ',
        #round(df2.transactions.mean(), 2) # avg trans per week for whole period
        # avg_item_per_trans=round((df2.item_sold[df2.week_no==17]/df2.transactions[df2.week_no==17]).mean(),2)#items per transactions w
        itemTrans=round((df2.item_sold / df2.transactions).mean(), 2)
        # max_w=round(max(df2.sales_amount[df2.week_no==17]),2)
        # min_w=round(min(df2.sales_amount[df2.week_no==17]),2)
        maxSale=round(max(df2.sales_amount), 2)
        minSale=round(min(df2.sales_amount), 2)
        # worst=df2.week_no[df2.sales_amount>min(df2.sales_amount)]
        #df2[['week_no', 'sales_amount']][(df2.sales_amount >= min(df2.sales_amount)) & (df2.sales_amount <= min(df2.sales_amount) + 1500)])
        #df2[['week_no', 'sales_amount']][(df2.sales_amount <= max(df2.sales_amount)) & (df2.sales_amount >= max(df2.sales_amount) - 3000)])
        #print('Price of trans ', round((df2.sales_amount / df2.transactions).mean(), 2))
        #print('Price of item ', round((df2.sales_amount / df2.item_sold).mean(), 2))
        itt = []
        trr = []
        sale = []
        for i3 in nullweeks:
            item = drr.item_sold[drr.week_no == i3].mean()
            trans = drr.transactions[drr.week_no == i3].mean()
            salee = drr.sales_amount[drr.week_no == i3].mean()
            itt.append(item)
            trr.append(trans)
            sale.append(salee)
        df_insert = {'sales_amountt': sale, 'ittem': itt, 'trans': trr, 'weeks_no': nullweeks}
        df_insert1 = pd.DataFrame(df_insert)
        forecastdf = dd1.fillna({'mean_item': df_insert1.ittem, 'mean_trans': df_insert1.trans, 'sales': df_insert1.sales_amountt})
        regr3 = linear_model.LinearRegression()
        X = forecastdf[['mean_item', 'mean_trans']]
        Y = forecastdf.sales
        regr3.fit(X, Y)
        y_predictionss = regr3.predict(X)
        y_predictionss1 = pd.DataFrame(y_predictionss)
        forecastdf['future_sales1'] = y_predictionss1.values
        f1 = forecastdf.set_index('date')
        frames1 = [df2.sales_amount, f1.future_sales1]
        join1 = pd.concat(frames1)
        detrend_sdata1 = signal.detrend(join1)
        trend1 = join1 - detrend_sdata1


        r1 = []
        for jj1 in pd.DataFrame(df2.index.year.values).drop_duplicates().index.values:
            sale_year1 = df2.sales_amount[str(int(pd.DataFrame(df2.index.year).drop_duplicates().loc[jj1]))].mean()
            r1.append(sale_year1)
        years1 = pd.DataFrame(df2.index.year).drop_duplicates().start_date.values
        holday1 = []
        for t1 in years1[0:len(years1) - 1]:
            h1 = df2.sales_amount[df2.week_no >= 50][str(t1)].mean() + df2.sales_amount[df2.week_no <= 3][str(int(t1) + 1)].mean()
            holday1.append(h1 / 2)
        year_last1 = f1.future_sales1[f1.weeks_no >= 50][str(years1[-1])].mean() + f1.future_sales1[f1.weeks_no <= 3].mean()  # 2018
        holday1.append(year_last1 / 2)
        N1 = len(r1)
        Holiday_Means1 = holday1
        All_Year_Means1 = r1
        ind1 = np.arange(N1)

        f_sale = int(forecastdf.future_sales1[forecastdf.weeks_no == WeekNo1].values)

        n_week = WeekNo1
        id_shop = shopID1
        sale_mean = avg_sale
        max_sale = maxSale
        min_sale = minSale
        item_trans = itemTrans
        # print(y.index)
        fig3, ax3 = plt.subplots(figsize=(7,4))
        # dates = pd.date_range(y.index[0], periods=104, freq='W-MON',format='%Y-%m-%d')
        # plt.plot(y.index,y,color='blue',label="actual sales")
        ax3.plot(y.index, a.sales_amount, color='red', label="actual sales")
        ax3.plot(dates, y_predictionss1, color='green', label="forecasted sales")
        ax3.plot(join1.index, trend1, color='blue', alpha=0.5, label='Trend')
        ax3.set_title('Comparison actual and predicted sales for whole period of shop ' + str(shopID1) + '\n')
        ax3.set_xlabel('Weeks')
        ax3.set_ylabel('Sales amount')
        ax3.legend()
        graph = mpld3.fig_to_html(fig3)


        fig4, ax4 = plt.subplots(figsize=(7,4))
        bar_width1 = 0.4
        opacity1 = 0.8
        ax4.bar(ind1, Holiday_Means1, bar_width1, opacity1, label='Holidays')
        ax4.bar(ind1 + bar_width1, All_Year_Means1, bar_width1, opacity1, label='Avg sales per year')

        ax4.set_ylabel('Sales_amount')
        ax4.set_title('Holiday sales (Xmas & NY) vs Average sales per year (shop #%s)' % shopID1)
        plt.xticks(ind1 + 0.25, (pd.DataFrame(df2.index.year).drop_duplicates().start_date.values))
        ax4.legend()
        graph1 = mpld3.fig_to_html(fig4)



    return render_template('index.html',graph1=graph1,graph=graph,value6=itemTrans,
                           value5=min_sale,value4=max_sale,value3=sale_mean,
                           value2=id_shop,value1=n_week,value=f_sale)
@app.route('/multinput/')
def multiget():
    return render_template ("multiinput.html")
@app.route('/multinput/',methods=['POST'])
def multi_input():
        shop_Id = request.form['shop_ids']
        shopID=[int(x) for x in shop_Id.split(',')]
        data = pd.read_csv('C:/weekly sales and labour cost for all shops 2013 to 20177.csv',
                           index_col='start_date', parse_dates=True)
        # shopID = input("Enter your shop id")
        shopID1 = list(shopID)
        for j in shopID1:

            data2 = data[['sales_id', 'shop_id', 'week_no', 'sales_amount', 'item_sold', 'transactions', 'total_tax',
                          'sales_status']]
            df1 = data2[data2.shop_id == j]  # input №1
            df2 = df1[df1.sales_status != 0]
            df2.week_no.isnull().values.any()
            nulldetect = df1.week_no.isnull()
            nulldetect[nulldetect == True].index
            df2.week_no.loc[nulldetect == True] = 54
            df2['week_no'] = df2.week_no - 2
            if len(df2.week_no) > 51:

                dff = df2[['sales_amount']]
                data3 = dff.reset_index()
                data4 = data3

                data5 = data4.rename(columns={'start_date': 'ds', 'sales_amount': 'y'})
                data5.set_index('ds')
                # y.plot()
                data5['y'] = np.log(data5['y'])
                data5 = data5.replace([np.inf, -np.inf], np.nan).fillna(0)
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

                #weekNO = int(input("Enter week number for shop %s" % j))

                print("Predicted sales amount for shop #" + str(j) + ": " + str(
                    float(predict_data1.future_sales[predict_data1.week_no == 23].values[0])))
                # fig, ax1 = plt.subplots()
                plt.plot(viz_df.sales_amount, label='Actual Sales shop %s' % j)
                plt.plot(viz_df.yhat_scaled, label='Forecasted Sales %s' % j)
                # ax1.fill_between(viz_df.index, np.exp(viz_df['yhat_upper']), np.exp(viz_df['yhat_lower']), alpha=0.5, color='darkgray')
                plt.title('Sales (Orange) vs Sales Forecast (Black) for shop ' + str(j))
                plt.ylabel('Dollar Sales')
                plt.xlabel('Dates')
                plt.legend()
            else:
                # dff4=df2.set_index('start_date')
                # df4=df3.week_no-2
                a = df2[['week_no', 'shop_id', 'sales_amount', 'transactions', 'total_tax', 'item_sold']]
                # start=[]
                # idd=list1


                dates = pd.date_range(df2.index[-1], periods=52, freq='W-MON', format='%Y-%m-%d')
                dates1 = pd.DataFrame(dates)
                dates2 = pd.date_range(df2.index[0], periods=len(df2.index), freq='W-MON', format='%Y-%m-%d')

                # dates1.set_index()
                mean_week_item = []
                for i in dates.week:
                    mean_item_sold = a.item_sold[a.week_no == i].mean()
                    mean_week_item.append(mean_item_sold)
                mean_week_item1 = pd.DataFrame(mean_week_item)

                trans_week_item = []
                for i1 in dates.week:
                    mean_trans_sold = a.transactions[a.week_no == i1].mean()
                    trans_week_item.append(mean_trans_sold)

                sales_week = []
                for ii1 in dates.week:
                    mean_sales_sold = a.sales_amount[a.week_no == ii1].mean()
                    sales_week.append(mean_sales_sold)

                dd = {'date': dates, 'weeks_no': dates.week, 'sales': sales_week, 'mean_item': mean_week_item,
                      'mean_trans': trans_week_item}
                dd1 = pd.DataFrame(dd)

                data1 = pd.read_csv(
                    'C:/weekly sales and labour cost for all shops 2013 to 20177.csv')

                dff1 = data1[data1.sales_status != 0]
                nulldetect = dff1.week_no.isnull()
                dff1.week_no.loc[nulldetect == True] = 54
                dff1['week_no'] = dff1.week_no - 2
                X_Cluster = dff1[['shop_id', 'sales_amount']]
                from sklearn.cluster import KMeans
                kmeans_model = KMeans(n_clusters=3, random_state=8).fit(X_Cluster)
                y_hat = kmeans_model.labels_  # clusters
                cen = kmeans_model.cluster_centers_
                y_hat1 = pd.DataFrame(y_hat)
                group_low_sales = X_Cluster[y_hat == 0]
                group_middle_sales = X_Cluster[y_hat == 2]
                group_high_sales = X_Cluster[y_hat == 1]

                fff = []
                for j in X_Cluster.shop_id:
                    dfdf = X_Cluster.sales_amount[X_Cluster.shop_id == j].mean()
                    fff.append(dfdf)
                f3 = pd.DataFrame(X_Cluster.shop_id.drop_duplicates())
                f4 = pd.DataFrame(fff)
                f5 = f4.drop_duplicates()
                f3['salle'] = f5.values

                # from sklearn.cluster import KMeans
                Xx2 = f3[['shop_id', 'salle']]
                kmeans_model2 = KMeans(n_clusters=3, random_state=8).fit(Xx2)
                y_hat2 = kmeans_model2.labels_  # clusters
                cen2 = kmeans_model2.cluster_centers_

                group_middle_sales2 = Xx2[y_hat2 == 0]
                group_high_sales2 = Xx2[y_hat2 == 2]
                group_low_sales2 = Xx2[y_hat2 == 1]

                # cx=int(group_low_sales.sales_amount[group_low_sales.shop_id==uu].values.mean())
                # trt=group_low_sales[group_low_sales.sales_amount>cx-3000]
                # trt2=trt[trt.sales_amount<cx+3000]


                nullweeks = dd1.weeks_no[dd1.mean_trans.isnull() == True]
                q = int(a.shop_id.mean())

                if (group_low_sales2.shop_id.values == q).any() == True:
                    cx = int(group_low_sales.sales_amount[group_low_sales.shop_id == q].values.mean())
                    trt = group_low_sales[group_low_sales.sales_amount > cx - 3000]
                    trt2 = trt[trt.sales_amount < cx + 3000]
                    valid_cls = dff1[['sales_amount', 'item_sold', 'transactions', 'week_no']].loc[trt2.index.values]
                    #print("Cluster of shop %s is low sales" % q)
                    # print("Average sales per week of shop %s is" %uu,cx)
                elif (group_middle_sales2.shop_id.values == q).any() == True:
                    # valid_cls=dff1[['sales_amount','item_sold','transactions','week_no']].loc[group_middle_sales.shop_id.index.values]
                    cx = int(group_middle_sales.sales_amount[group_middle_sales.shop_id == q].values.mean())
                    trt = group_middle_sales[group_middle_sales.sales_amount > cx - 3000]
                    trt2 = trt[trt.sales_amount < cx + 3000]
                    valid_cls = dff1[['sales_amount', 'item_sold', 'transactions', 'week_no']].loc[trt2.index.values]
                    #print("Cluster of shop %s is average sales" % q)
                    # print("Average sales per week of shop %s is " %uu,cx)
                elif (group_high_sales2.shop_id.values == q).any() == True:
                    # valid_cls=dff1[['sales_amount','item_sold','transactions','week_no']].loc[group_high_sales.shop_id.index.values]
                    cx = int(group_high_sales.sales_amount[group_high_sales.shop_id == q].values.mean())
                    trt = group_high_sales[group_high_sales.sales_amount > cx - 4000]
                    trt2 = trt[trt.sales_amount < cx + 4000]
                    valid_cls = dff1[['sales_amount', 'item_sold', 'transactions', 'week_no']].loc[trt2.index.values]
                    #print("Cluster of shop %s is high sales" % q)
                    # print("Average sales per week of shop %s is" %uu,cx)
                    # drr=valid_cls


                    # if (group_low_sales2.shop_id.values==99).any()==True:
                    # valid_cls=dff1[['sales_amount','item_sold','transactions','week_no']].loc[group_low_sales.shop_id.index.values]
                    # elif (group_middle_sales2.shop_id.values==99).any()==True:
                    # valid_cls=dff1[['sales_amount','item_sold','transactions','week_no']].loc[group_middle_sales.shop_id.index.values]
                    # elif (group_high_sales2.shop_id.values==99).any()==True:
                    # valid_cls=dff1[['sales_amount','item_sold','transactions','week_no']].loc[group_high_sales.shop_id.index.values]

                drr = valid_cls  # dff1[['sales_amount','item_sold','transactions','week_no']].loc[trt2.index.values]
                itt = []
                trr = []
                sale = []
                for i3 in nullweeks:
                    item = drr.item_sold[drr.week_no == i3].mean()
                    trans = drr.transactions[drr.week_no == i3].mean()
                    salee = drr.sales_amount[drr.week_no == i3].mean()
                    itt.append(item)
                    trr.append(trans)
                    sale.append(salee)
                df_insert = {'sales_amountt': sale, 'ittem': itt, 'trans': trr, 'weeks_no': nullweeks}
                df_insert1 = pd.DataFrame(df_insert)
                # group_low_sales.shop_id.drop_duplicates().index.values
                # null=dd1.isnull()
                # dd1.isnull().loc[null==True]

                # for i4 in dates.week:
                # a.transactions[a.week_no==i1].mean()
                # trans_week_item.append(mean_trans_sold)
                forecastdf = dd1.fillna(
                    {'mean_item': df_insert1.ittem, 'mean_trans': df_insert1.trans, 'sales': df_insert1.sales_amountt})
                # forecastdf
                # print("Average amount of transactions per week of shop %s is " %uu+str(int(forecastdf.mean_trans.mean()))+"\n")
                regr3 = linear_model.LinearRegression()
                X = forecastdf[['mean_item', 'mean_trans']]
                Y = forecastdf.sales
                regr3.fit(X, Y)
                y_predictionss = regr3.predict(X)
                y_predictionss1 = pd.DataFrame(y_predictionss)
                # dff1[['item_sold','transactions','week_no']].index#group_low_sales.shop_id.drop_duplicates().index
                # plt.figure(figsize=(19,6))
                # from pylab import rcParams
                #plt.rcParams['figure.figsize'] = 15, 10
                #from pylab import rcParams
                #rcParams['figure.figsize'] = (20, 10)
                plt.plot(dates2, df2.sales_amount, label="actual sales shop %s" % q)
                plt.plot(dates, y_predictionss1, label="predicted sales shop %s" % q)
                plt.title('Comparison actual and predicted sales for whole period of shops %s' %shopID1)
                plt.xlabel('Weeks')
                plt.ylabel('Sales amount')
                plt.legend()
                figg = plt.gcf()
                figg.set_size_inches(13, 7)
                # print(valid_cls)

        #mpld3.show()


                # L=ax1.legend() #get the legend
                # L.get_texts()[0].set_text('Actual Sales') #change the legend text for 1st plot
                # L.get_texts()[1].set_text('Forecasted Sales') #change the legend text for 2nd plot

        #plt.show()
        #img = BytesIO()
        #plt.savefig(img, format='png')
        #img.seek(0)
        #return send_file(img, mimetype='image/png')
        #figg=mpld3
        graphh = mpld3.fig_to_html(figg)
        #mpld3.show(fig)
#@app.route('/multi/')
#def multi_input1():
        return render_template('multinput.html',graphh=graphh,value2=shopID1)

#fan hui json fang fa
@app.route('/tablejson', methods=['GET', 'POST'])
def tablejson():
    import codecs
    fd = codecs.open('df.json')
    df=fd.read()
    fd.close()
    print(df)
    return df

#download function
from flask import send_file, send_from_directory
import os
@app.route("/download/<filename>", methods=['GET'])
def download_file(filename):
    # 需要知道2个参数, 第1个参数是本地目录的path, 第2个参数是文件名(带扩展名)
    directory = os.getcwd()  # 假设在当前目录
    print('down')
    return send_from_directory(directory, filename, as_attachment=True)



@app.route('/df_tabb', methods=['GET', 'POST'])
def memval2():
    print(memval2)
    return memval2




#models


#Table generator

def Table_generator():
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    from pylab import rcParams
    from sklearn import linear_model
    from fbprophet import Prophet
    from numpy import inf
    filepath = 'C:/weekly sales and labour cost for all shops 2013 to 20177.csv'
    df = pd.read_csv(filepath)
    df2 = df[df.sales_status != 0]
    # df2.week_no.isnull().values.any()
    nulldetect = df2.week_no.isnull()
    # nulldetect[nulldetect==True].index
    df2.week_no.loc[nulldetect == True] = 54
    df2['week_no'] = df2.week_no - 2
    len_week1 = []
    for i in df2.shop_id:
        len_week = len(df2.week_no[df2.shop_id == i])
        len_week1.append(len_week)
    len_week2 = pd.DataFrame(len_week1)
    len_week2
    d = {'shop_id': df2.shop_id, 'len_of_weeks': len_week1}
    d1 = pd.DataFrame(d)
    d2 = d1.drop_duplicates()

    dtt = pd.DataFrame(index=list(range(1, 53)), columns=d2.shop_id[d2.len_of_weeks < 52].values)
    # table = pd.DataFrame(columns=['shop_id','week_no','dates','forecasted_sales'])

    for uu in d2.shop_id[d2.len_of_weeks < 52].values:
        df3 = df2[df2.shop_id == uu]
        # df1 = df[(= -1) & (df.b != -1)]
        dff4 = df3.set_index('start_date')
        # df4=df3.week_no-2
        a = df3[['week_no', 'shop_id', 'sales_amount', 'transactions', 'total_tax', 'item_sold']]
        # print(a)
        dates = pd.date_range(dff4.index[-1], periods=52, freq='W-MON', format='%Y-%m-%d')
        dates1 = pd.DataFrame(dates)
        dates2 = pd.date_range(dff4.index[0], periods=len(dff4.index), freq='W-MON', format='%Y-%m-%d')
        mean_week_item = []
        for i in dates.week:
            mean_item_sold = a.item_sold[a.week_no == i].mean()
            mean_week_item.append(mean_item_sold)
        mean_week_item1 = pd.DataFrame(mean_week_item)

        trans_week_item = []
        for i1 in dates.week:
            mean_trans_sold = a.transactions[a.week_no == i1].mean()
            trans_week_item.append(mean_trans_sold)

        sales_week = []
        for ii1 in dates.week:
            mean_sales_sold = a.sales_amount[a.week_no == ii1].mean()
            sales_week.append(mean_sales_sold)
        dd = {'date': dates, 'weeks_no': dates.week, 'sales': sales_week, 'mean_item': mean_week_item,
              'mean_trans': trans_week_item}
        dd1 = pd.DataFrame(dd)
        dff1 = df[df.sales_status != 0]
        nulldetect = dff1.week_no.isnull()
        dff1.week_no.loc[nulldetect == True] = 54
        dff1['week_no'] = dff1.week_no - 2
        X_Cluster = dff1[['shop_id', 'sales_amount']]
        from sklearn.cluster import KMeans
        kmeans_model = KMeans(n_clusters=3, random_state=8).fit(X_Cluster)
        y_hat = kmeans_model.labels_  # clusters
        cen = kmeans_model.cluster_centers_
        y_hat1 = pd.DataFrame(y_hat)
        group_low_sales = X_Cluster[y_hat == 0]
        group_middle_sales = X_Cluster[y_hat == 2]
        group_high_sales = X_Cluster[y_hat == 1]
        fff = []
        for j in X_Cluster.shop_id:
            dfdf = X_Cluster.sales_amount[X_Cluster.shop_id == j].mean()
            fff.append(dfdf)
        f3 = pd.DataFrame(X_Cluster.shop_id.drop_duplicates())
        f4 = pd.DataFrame(fff)
        f5 = f4.drop_duplicates()
        f3['salle'] = f5.values

        Xx2 = f3[['shop_id', 'salle']]
        kmeans_model2 = KMeans(n_clusters=3, random_state=8).fit(Xx2)
        y_hat2 = kmeans_model2.labels_  # clusters
        cen2 = kmeans_model2.cluster_centers_

        group_middle_sales2 = Xx2[y_hat2 == 0]
        group_high_sales2 = Xx2[y_hat2 == 2]
        group_low_sales2 = Xx2[y_hat2 == 1]
        nullweeks = dd1.weeks_no[dd1.mean_trans.isnull() == True]

        if (group_low_sales2.shop_id.values == uu).any() == True:
            cx = int(group_low_sales.sales_amount[group_low_sales.shop_id == uu].values.mean())
            trt = group_low_sales[group_low_sales.sales_amount > cx - 3000]
            trt2 = trt[trt.sales_amount < cx + 3000]
            valid_cls = dff1[['sales_amount', 'item_sold', 'transactions', 'week_no']].loc[trt2.index.values]
            # print("Cluster of shop %s is low sales" %uu)
            # print("Average sales per week of shop %s is" %uu,cx)
        elif (group_middle_sales2.shop_id.values == uu).any() == True:
            # valid_cls=dff1[['sales_amount','item_sold','transactions','week_no']].loc[group_middle_sales.shop_id.index.values]
            cx = int(group_middle_sales.sales_amount[group_middle_sales.shop_id == uu].values.mean())
            trt = group_middle_sales[group_middle_sales.sales_amount > cx - 3000]
            trt2 = trt[trt.sales_amount < cx + 3000]
            valid_cls = dff1[['sales_amount', 'item_sold', 'transactions', 'week_no']].loc[trt2.index.values]
            # print("Cluster of shop %s is average sales" %uu)
            # print("Average sales per week of shop %s is " %uu,cx)
        elif (group_high_sales2.shop_id.values == uu).any() == True:
            # valid_cls=dff1[['sales_amount','item_sold','transactions','week_no']].loc[group_high_sales.shop_id.index.values]
            cx = int(group_high_sales.sales_amount[group_high_sales.shop_id == uu].values.mean())
            trt = group_high_sales[group_high_sales.sales_amount > cx - 4000]
            trt2 = trt[trt.sales_amount < cx + 4000]
            valid_cls = dff1[['sales_amount', 'item_sold', 'transactions', 'week_no']].loc[trt2.index.values]
            # print("Cluster of shop %s is high sales" %uu)
            # print("Average sales per week of shop %s is" %uu,cx)
        drr = valid_cls
        drr = valid_cls  # dff1[['sales_amount','item_sold','transactions','week_no']].loc[trt2.index.values]
        itt = []
        trr = []
        sale = []
        for i3 in nullweeks:
            item = drr.item_sold[drr.week_no == i3].mean()
            trans = drr.transactions[drr.week_no == i3].mean()
            salee = drr.sales_amount[drr.week_no == i3].mean()
            itt.append(item)
            trr.append(trans)
            sale.append(salee)
        df_insert = {'sales_amountt': sale, 'ittem': itt, 'trans': trr, 'weeks_no': nullweeks}
        df_insert1 = pd.DataFrame(df_insert)
        forecastdf = dd1.fillna(
            {'mean_item': df_insert1.ittem, 'mean_trans': df_insert1.trans, 'sales': df_insert1.sales_amountt})
        forecastdf1 = forecastdf.fillna({'mean_item': df_insert1.ittem.mean(),
                                         'mean_trans': df_insert1.trans.mean(),
                                         'sales': df_insert1.sales_amountt.mean()})
        regr3 = linear_model.LinearRegression()
        X = forecastdf1[['mean_item', 'mean_trans']]
        Y = forecastdf1.sales
        regr3.fit(X, Y)
        y_predictionss = regr3.predict(X)
        y_predictionss1 = pd.DataFrame(y_predictionss)
        pred_y = round(y_predictionss1, 2)
        #print(pred_y.values)
        forecastdf1['forecasted_sales'] = pred_y.values
        # ddt.fillna()
        forecastdf1.sort_values('weeks_no', inplace=True)
        # forecastdf1
        # forecastdf1.forecasted_sales.reset_index()
        f = forecastdf1.set_index('weeks_no')
        # dtt = pd.DataFrame(index=list(range(1,53)), columns=d2.shop_id[d2.len_of_weeks<52].values)
        dtt['shop_id'] = dtt.index.values
        # dtt[dtt.shop_id==uu].fillna()
        dtt[[uu]] = f.forecasted_sales.values.reshape((52, 1))

    dtt1 = pd.DataFrame(index=d2.shop_id[d2.len_of_weeks < 52].values, columns=list(range(1, 53)))
    for jj in dtt.index.values:
        dtt1.loc[:, jj] = dtt.loc[jj, :]

    data = pd.read_csv('C:/weekly sales and labour cost for all shops 2013 to 20177.csv', index_col='start_date',
                       parse_dates=True)
    # shopID = input("Enter your shop id")

    df2 = data[data.sales_status != 0]
    # df2.week_no.isnull().values.any()
    nulldetect = df2.week_no.isnull()
    # nulldetect[nulldetect==True].index
    df2.week_no.loc[nulldetect == True] = 54
    df2['week_no'] = df2.week_no - 2

    len_week1 = []
    for i in df2.shop_id:
        len_week = len(df2.week_no[df2.shop_id == i])
        len_week1.append(len_week)
    len_week2 = pd.DataFrame(len_week1)
    len_week2
    d = {'shop_id': df2.shop_id, 'len_of_weeks': len_week1}
    d1 = pd.DataFrame(d)
    d2 = d1.drop_duplicates()

    dtt2 = pd.DataFrame(index=list(range(1, 53)), columns=d2.shop_id[d2.len_of_weeks > 52].values)

    for j in d2.shop_id[d2.len_of_weeks >= 52].values:
        data2 = data[['sales_id', 'shop_id', 'week_no', 'sales_amount', 'item_sold', 'transactions', 'total_tax',
                      'sales_status']]
        df1 = data2[data2.shop_id == j]  # input №1
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
        data5 = data5.replace([np.inf, -np.inf], np.nan).fillna(0)
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
        predict_data = {'shop_id': int(df2.shop_id.mean()), 'future_sales': future_sales1, 'week_no': week_no,
                        'start_date': start_date}
        predict_data1 = pd.DataFrame(predict_data)
        predict_data1 = predict_data1.drop_duplicates(subset=['week_no'])
        predict_data1.sort_values('week_no', inplace=True)
        f1 = predict_data1.set_index('week_no')
        dtt2[[j]] = f1.future_sales.values.reshape((52, 1))

    dtt3 = pd.DataFrame(index=d2.shop_id[d2.len_of_weeks > 52].values, columns=list(range(1, 53)))
    for qq in dtt.index.values:
        dtt3.loc[:, qq] = dtt2.loc[qq, :]

    tab = dtt1.append(dtt3)
    tab['shop_id'] = tab.index.values
    tab.sort_values('shop_id', inplace=True)
    tab_id = tab.shop_id
    tab = tab.drop('shop_id', axis=1)
    tab.insert(0, 'shop_id', tab_id)
    #writer = pd.ExcelWriter('output.xlsx')
    #tab.to_excel(writer, 'Sheet1')
    #writer.save()
    tab.to_json(path_or_buf='df.json', orient='records')
    memval2 =tab
#Table_generator()
    #print('==============table complet=================')
@app.route('/table/')
def table():
    return render_template('maindashboard.html')

if __name__ == '__main__':
    app.run()
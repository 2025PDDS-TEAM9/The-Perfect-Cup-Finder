import sqlite3
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback

# connect db and get data
conn = sqlite3.connect('Tea_Shop_Database.db')
cursor = conn.cursor()

# get all drink names
query_drink = 'SELECT drink_name FROM Drink'
cursor.execute(query_drink)
data_drink = cursor.fetchall()

# get ingredient information for each drink
query_ingr = '''
SELECT D.drink_name, B.base_name, B.base_price, BI.ingr_ml, I.ingr_name, I.ingr_cal, I.ingr_caff
FROM Drink AS D
JOIN Base AS B ON D.base_id = B.base_id
JOIN BaseIngredient AS BI ON B.base_id = BI.base_id
JOIN Ingredient AS I ON BI.ingr_id = I.ingr_id
'''
cursor.execute(query_ingr)
data_ingr = cursor.fetchall()

# get topping information for each drink
query_tp = '''
SELECT D.drink_name, T.tp_name, T.tp_price, T.tp_cal, T.tp_caff
FROM Drink AS D
LEFT JOIN DrinkTopping AS DT ON D.drink_id = DT.drink_id
LEFT JOIN Topping AS T ON DT.tp_id = T.tp_id
'''
cursor.execute(query_tp)
data_tp = cursor.fetchall()

# get sweetness information for each drink
# query_sw = '''SELECT E.fname || ' ' || E.lname, A.mname || ' ' || A.model, E.salary
#            FROM Employee AS E
#            LEFT JOIN Certificate AS C ON E.eid = C.eid
#            LEFT JOIN Aircraft AS A ON C.aid = A.aid
#            '''
# cursor.execute(query_sw)
# data_sw = cursor.fetchall()

conn.close()

# style definition
# 底色 (奶茶色) --> #EBDEC1
# 標題 (深咖啡)  --> #5C4033
# 白色底框 --> white

# data preprocessing
drink_list = [item[0] for item in data_drink]
df_ingr = pd.DataFrame(data_ingr, columns = ['drink_name', 'base_name', 'base_price', 'ingr_ml', 'ingr_name', 'ingr_cal', 'ingr_caff'])
df_tp = pd.DataFrame(data_tp, columns = ['drink_name', 'tp_name', 'tp_price', 'tp_cal', 'tp_caff'])
df_tp['tp_name'] = df_tp['tp_name'].fillna('NULL')
df_tp['tp_price'] = df_tp['tp_price'].fillna(0)
df_tp['tp_cal'] = df_tp['tp_cal'].fillna(0)
df_tp['tp_caff'] = df_tp['tp_caff'].fillna(0)
drink_info = pd.DataFrame({'drink_name': drink_list})
# dashboard
# get price (base_price + tp_price)
price_list = []
for drink in drink_list:
    base_price = df_ingr.loc[df_ingr['drink_name'] == drink, 'base_price'].iloc[0]
    tp_price = df_tp.loc[df_tp['drink_name'] == drink, 'tp_price'].iloc[0]
    price_list.append(base_price + tp_price)
drink_info['drink_price'] = price_list

# # get cal (all ingr_cal + tp_cal)
cal_list = []
for drink in drink_list:
    # calculate total ingr_cal
    base_cal = 0
    trim_df = df_ingr.loc[df_ingr['drink_name'] == drink]
    for idx, row in trim_df.iterrows():
        base_cal += (row['ingr_cal'] * row['ingr_ml'] / 100)
    tp_cal = df_tp.loc[df_tp['drink_name'] == drink, 'tp_cal'].iloc[0]
    cal_list.append(base_cal + tp_cal)
drink_info['drink_cal'] = cal_list

# get caff (all ingr_caff + tp_caff)
caff_list = []
for drink in drink_list:
    # calculate total ingr_caff
    base_caff = 0
    trim_df = df_ingr.loc[df_ingr['drink_name'] == drink]
    for idx, row in trim_df.iterrows():
        base_caff += (row['ingr_caff'] * row['ingr_ml'] / 100)
    tp_caff = df_tp.loc[df_tp['drink_name'] == drink, 'tp_caff'].iloc[0]
    caff_list.append(base_caff + tp_caff)
drink_info['drink_caff'] = caff_list
print(drink_info)



# note
# 飲料的ingr: 
# (drink_id), drink_name, (base_id), base_name, base_price, (ingr_id), ingr_ml, ingr_name, ingr_cal, ingr_caff

# 飲料的topping:
# (drink_id), drink_name, (tp_id), tp_name, tp_price, tp_cal, tp_caff

# 飲料的甜度:
# (drink_id), drink_name, (sw_id), sw_name, sw_cal
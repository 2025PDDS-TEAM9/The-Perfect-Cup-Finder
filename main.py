import sqlite3
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback

# --- 1. 連接資料庫並獲取資料 ---
conn = sqlite3.connect('Tea_Shop_Database.db')
cursor = conn.cursor()

# 取得所有飲料名稱
query_drink = 'SELECT drink_name FROM Drink'
cursor.execute(query_drink)
data_drink = cursor.fetchall()

# 取得每種飲料的成分資訊
query_ingr = '''
SELECT D.drink_name, B.base_name, B.base_price, BI.ingr_ml, I.ingr_name, I.ingr_cal, I.ingr_caff
FROM Drink AS D
JOIN Base AS B ON D.base_id = B.base_id
JOIN BaseIngredient AS BI ON B.base_id = BI.base_id
JOIN Ingredient AS I ON BI.ingr_id = I.ingr_id
'''
cursor.execute(query_ingr)
data_ingr = cursor.fetchall()

# 取得每種飲料的配料資訊
query_tp = '''
SELECT D.drink_name, T.tp_name, T.tp_price, T.tp_cal, T.tp_caff
FROM Drink AS D
LEFT JOIN DrinkTopping AS DT ON D.drink_id = DT.drink_id
LEFT JOIN Topping AS T ON DT.tp_id = T.tp_id
'''
cursor.execute(query_tp)
data_tp = cursor.fetchall()
conn.close()

drink_list = [x[0] for x in data_drink]

# 建立成分的表格 (DataFrame)
df_ingr = pd.DataFrame(data_ingr, columns=['drink_name', 'base_name', 'base_price', 'ingr_ml', 'ingr_name', 'ingr_cal', 'ingr_caff'])

# 建立配料的表格 (DataFrame)
df_tp = pd.DataFrame(data_tp, columns=['drink_name', 'tp_name', 'tp_price', 'tp_cal', 'tp_caff'])

df_tp = df_tp.fillna(0)

drink_info = pd.DataFrame({'drink_name': drink_list})

# --- 2. 計算數值 ---

# 計算價格 (基底價格 + 配料價格)
price_list = []
for drink in drink_list:
    base_price = df_ingr.loc[df_ingr['drink_name'] == drink, 'base_price'].iloc[0]
    tp_price = df_tp.loc[df_tp['drink_name'] == drink, 'tp_price'].iloc[0]
    price_list.append(base_price + tp_price)
drink_info['drink_price'] = price_list

# 計算熱量 (所有成分熱量 + 配料熱量)
cal_list = []
for drink in drink_list:
    base_cal = 0
    trim_df = df_ingr.loc[df_ingr['drink_name'] == drink]
    for idx, row in trim_df.iterrows():
        base_cal += (row['ingr_cal'] * row['ingr_ml'] / 100)
    tp_cal = df_tp.loc[df_tp['drink_name'] == drink, 'tp_cal'].iloc[0]
    cal_list.append(base_cal + tp_cal)
drink_info['drink_cal'] = cal_list

# 計算咖啡因 (所有成分咖啡因 + 配料咖啡因)
caff_list = []
for drink in drink_list:
    base_caff = 0
    trim_df = df_ingr.loc[df_ingr['drink_name'] == drink]
    for idx, row in trim_df.iterrows():
        base_caff += (row['ingr_caff'] * row['ingr_ml'] / 100)
    tp_caff = df_tp.loc[df_tp['drink_name'] == drink, 'tp_caff'].iloc[0]
    caff_list.append(base_caff + tp_caff)
drink_info['drink_caff'] = caff_list

# --- 3. 建立 Dash App ---

app = Dash(__name__)

# 設定網頁的外觀 (Layout)
app.layout = html.Div([
    # 修改大標題顏色 
    html.H1("Start Your Drink Journey: What's Trending?", 
            style={'color': '#5C4033', 'textAlign': 'center', 'paddingTop': '20px'}),

    # 放置控制元件的區塊
    html.Div([
        html.Label("選擇比較項目:", style={'color': '#5C4033', 'fontWeight': 'bold'}), # 標籤也改深色
        dcc.Dropdown(
            id='metric-dropdown',
            options=[
                {'label': 'Kcal (熱量)', 'value': 'drink_cal'},
                {'label': 'Price (價格)', 'value': 'drink_price'},
                {'label': 'Caffeine (咖啡因)', 'value': 'drink_caff'}
            ],
            value='drink_cal',
            clearable=False,
            style={'width': '200px', 'marginLeft': '10px'}
        ),
        
        html.Label("排序方式:", style={'marginLeft': '20px', 'color': '#5C4033', 'fontWeight': 'bold'}), # 標籤也改深色
        dcc.RadioItems(
            id='sort-order',
            options=[
                {'label': '前 10 名 (最高)', 'value': 'descending'},
                {'label': '最後 10 名 (最低)', 'value': 'ascending'}
            ],
            value='descending',
            style={'marginLeft': '10px', 'display': 'inline-block', 'color': '#5C4033'} # 選項文字也改深色
        )
    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'marginBottom': '20px', 'marginTop': '20px'}),

    # 放置圖表的空位
    dcc.Graph(id='bar-chart')

# 修改最外層背景色
], style={'backgroundColor': '#EBDEC1', 'minHeight': '100vh', 'padding': '20px'})

@callback(
    Output('bar-chart', 'figure'),
    Input('metric-dropdown', 'value'),
    Input('sort-order', 'value')
)
def update_graph(selected_metric, sort_order):
    # 根據選擇的項目排序
    is_ascending = True if sort_order == 'ascending' else False
    
    sorted_df = drink_info.sort_values(by=selected_metric, ascending=is_ascending)
    
    # 取出前 10 筆資料
    top_10_df = sorted_df.head(10)
    

    labels_map = {
        'drink_cal': '熱量 (Kcal)',
        'drink_price': '價格 (NTD)',
        'drink_caff': '咖啡因 (mg)',
        'drink_name': '飲料名稱'
    }

    # 3. 畫圖
    fig = px.bar(
        top_10_df,
        x='drink_name',         
        y=selected_metric,      
        text=selected_metric,   
        labels=labels_map,      
        title=f"飲料{labels_map[selected_metric]}排名 ({'最低' if is_ascending else '最高'} 10 名)"
    )
    
    
    fig.update_traces(textposition='outside', marker_color='#5C4033') # 我順便把柱子的顏色也改成深咖啡色，看看你喜不喜歡

    fig.update_layout(
        xaxis={'categoryorder':'total ascending'} if is_ascending else {'categoryorder':'total descending'},
        height=600,
        
        # --- 顏色的設定 ---
        plot_bgcolor='white',      
        paper_bgcolor='#EBDEC1',    
        font_color='#5C4033',       
        title_font_color='#5C4033', 
        # -------------------------
        margin=dict(l=100, r=100, t=100, b=100) 
    )
    
    return fig

if __name__ == '__main__':
    app.run(debug=True)
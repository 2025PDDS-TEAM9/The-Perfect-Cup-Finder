import sqlite3
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback, State, dash_table
from dash import callback_context


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
query_sw = '''
SELECT D.drink_name, S.sw_name, S.sw_cal
FROM Drink AS D
LEFT JOIN DrinkSweetness AS DS ON D.drink_id = DS.drink_id
LEFT JOIN Sweetness AS S ON DS.sw_id = S.sw_id
'''
cursor.execute(query_sw)
data_sw = cursor.fetchall()

ingredients = pd.read_sql_query("SELECT ingr_id, ingr_name FROM Ingredient", conn)

conn.close()

# style definition
COLOR_LIGHT_BROWN = '#EBDEC1'
COLOR_DARK_BROWN = '#5C4033'
COLOR_WHITE = '#FFFFFF'
COLOR_BLACK = '#000000'
BORDER = False
def pic(name):
    return html.Img(src = f'/assets/{name}.png',
                    id = 'drink_pic',
                    style = {'width': '70%', 'marginBottom': '2vw'})

def to_pic(drink):
    file = drink.split(' (')[0].replace("'", '').replace(' ', '_').lower()
    return f'/assets/{file}.png'

def find_tp(drink):
    for item in tp_list:
        if item in drink:
            return item
    return 'NULL'

def block_style(mode):
    # if mode == 'full':
    # else:  # half
    return {'backgroundColor': COLOR_WHITE,
            'borderRadius': '20px',
            # 'padding': '0vw 5vw',  # 3vw
            'margin': '5vw',
            'height': '70vh'}

def word_style(size, alignment):
    return {'color': COLOR_DARK_BROWN,
            'fontFamily': 'Arial, sans-serif',
            'fontSize': f'{size}vw',
            'fontWeight': 'normal',  # bold, normal, 100~900
            'textAlign': alignment}

def dd_style():
    return {'color': COLOR_BLACK,
            'fontFamily': 'Arial, sans-serif',
            'fontSize': '1vw',
            'fontWeight': 'normal',  # bold, normal, 100~900
            'textAlign': 'left'}

def put_vertical(alignment):
    return {'display': 'flex', 'flexDirection': 'column', 'alignItems': alignment}

def put_horizontal(alignment):
    return {'display': 'flex', 'flexDirection': 'row', 'alignItems': alignment}

def plot_grid(title):
    return dict(
                title = title,
                showgrid = True,
                gridcolor = COLOR_LIGHT_BROWN,
                gridwidth = 1,
                zeroline = True,
                zerolinecolor = COLOR_LIGHT_BROWN,
                zerolinewidth = 2
            )

def add_border():
    if BORDER:
        return {'border': '5px dotted red'}
    else:
        return {}

# data preprocessing
drink_list = [item[0] for item in data_drink]
df_ingr = pd.DataFrame(data_ingr, columns = ['drink_name', 'base_name', 'base_price', 'ingr_ml', 'ingr_name', 'ingr_cal', 'ingr_caff'])
df_tp = pd.DataFrame(data_tp, columns = ['drink_name', 'tp_name', 'tp_price', 'tp_cal', 'tp_caff'])
df_tp['tp_name'] = df_tp['tp_name'].fillna('NULL')
df_tp['tp_price'] = df_tp['tp_price'].fillna(0)
df_tp['tp_cal'] = df_tp['tp_cal'].fillna(0)
df_tp['tp_caff'] = df_tp['tp_caff'].fillna(0)
df_sw = pd.DataFrame(data_sw, columns = ['drink_name', 'sw_name', 'sw_cal'])
drink_info_no_sugar = pd.DataFrame({'drink_name': drink_list})
tp_list = df_tp['tp_name'].unique().tolist()

# get base and price (base_price + tp_price)
price_list = []
base_list = []
for drink in drink_list:
    base_list.append(df_ingr.loc[df_ingr['drink_name'] == drink, 'base_name'].iloc[0])
    base_price = df_ingr.loc[df_ingr['drink_name'] == drink, 'base_price'].iloc[0]
    tp_price = df_tp.loc[df_tp['drink_name'] == drink, 'tp_price'].iloc[0]
    price_list.append(base_price + tp_price)
drink_info_no_sugar['drink_price'] = price_list
drink_info_no_sugar['drink_base'] = base_list

# get cal (all ingr_cal + tp_cal)
cal_list = []
for drink in drink_list:
    # calculate total ingr_cal
    base_cal = 0
    trim_df = df_ingr.loc[df_ingr['drink_name'] == drink]
    for idx, row in trim_df.iterrows():
        base_cal += (row['ingr_cal'] * row['ingr_ml'] / 100)
    tp_cal = df_tp.loc[df_tp['drink_name'] == drink, 'tp_cal'].iloc[0]
    cal_list.append(base_cal + tp_cal)
drink_info_no_sugar['drink_cal'] = cal_list

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
drink_info_no_sugar['drink_caff'] = caff_list

# get all drink_combination
all_drink_list_full = []
all_drink_list = []
all_base_list = []
all_price_list = []
all_cal_list = []
all_caff_list = []
for idx, row in df_sw.iterrows():
    all_drink_list_full.append(f"{row['drink_name']} ({row['sw_name']})")
    cor_row = drink_info_no_sugar.loc[drink_info_no_sugar['drink_name'] == row['drink_name'],
                                      ['drink_name', 'drink_base', 'drink_price', 'drink_cal', 'drink_caff']].iloc[0]
    all_drink_list.append(cor_row['drink_name'])
    all_base_list.append(cor_row['drink_base'])
    all_price_list.append(cor_row['drink_price'])
    all_cal_list.append(cor_row['drink_cal'] + row['sw_cal'])
    all_caff_list.append(cor_row['drink_caff'])

drink_info = pd.DataFrame({'drink_full_name': all_drink_list_full,
                           'drink_name': all_drink_list,
                           'drink_base': all_base_list,
                           'drink_price': all_price_list,
                           'drink_cal': all_cal_list,
                           'drink_caff': all_caff_list})

# dashboard
sw_dd = dcc.Dropdown(
            id = 'sw_dd',
            options = [
                {'label': 'Full Sugar (100%)', 'value': 'Full Sugar'},
                {'label': 'Half Sugar (50%)', 'value': 'Half Sugar'},
                {'label': 'Less Sugar (20%)', 'value': 'Less Sugar'},
                {'label': 'No Sugar (0%)', 'value': 'No Sugar'}
            ],
            value = 'Select a sweetness',
            style = {**dd_style(), 'marginBottom': '1vw'}
        )
base_dd = dcc.Dropdown(id = 'base_dd', options = base_list, 
                       value = 'Select a base', style = {**dd_style(), 'marginBottom': '1vw'})
tp_dd = dcc.Dropdown(id = 'tp_dd', options = tp_list,
                     value = 'Select a topping', style = dd_style())
scatter_fig = px.scatter(
  data_frame = drink_info,
  x = 'drink_cal',
  y = 'drink_price',
  color = 'drink_name',
  hover_data = ['drink_full_name'],
  custom_data = ['drink_full_name'],
  range_x = [0, 850],
  range_y = [20, 85]
#   title = 'Relationship Between Salary and Certificate Counts'
)
scatter_fig.update_layout(
    font = dict(family = 'Arial, sans-serif', size = 16, color = COLOR_DARK_BROWN),
    plot_bgcolor = COLOR_WHITE,
    xaxis = plot_grid('Calories'),
    yaxis = plot_grid('Price')
)
scatter_plot = dcc.Graph(
		    id = 'scatter_plot',
		    figure = scatter_fig,
		    style = {'width': '100%', 'height': '100%'}
            )

app = Dash()

app.layout = html.Div(
        children = [
            # dashboard 1
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
    dcc.Graph(id='bar-chart'),

            # dashboard 2
# 標題
        html.Div([
            html.H3('Choose Your Flavor Base:', 
                    style={'margin': '0', 'color': '#5C4033', 'fontWeight': 'bold'}),
            html.H4('Discover Drinks Made Just for You', 
                    style={'margin': '5px 0 0 0', 'color': '#5C4033', 'fontWeight': 'bold'}),
        ], style={'marginBottom': '25px'}),
            
        # 下拉選單
        html.Div([
            html.Label('Select Ingredient:', 
                        style={'fontWeight': '500', 'marginBottom': '8px', 'display': 'block', 'color': '#5C4033'}),
            dcc.Dropdown(
                id='ingredient',
                options=[{'label': row['ingr_name'], 'value': row['ingr_id']} 
                         for _, row in ingredients.iterrows()],
                placeholder='All...',
                style={
                    'width': '250px',  
                    'border': '1px solid #8B6F47', 
                    'borderRadius': '5px'
                }
            ),
        ], style={'marginBottom': '25px'}),

        # 內層（白色部分）
        html.Div([            
            html.Div(id='output')
        ], style={
            'backgroundColor': 'white',
            'padding': '35px',
            'borderRadius': '15px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
            'minHeight': '400px'
        }),

            # dashboard 3
            # memory for base and topping options
            dcc.Store(id = 'base_options_store', data = pd.Series(base_list).unique().tolist()),
            dcc.Store(id = 'tp_options_store', data = tp_list),

            # dropdown
            html.Div(children = [
                html.Div('Sweetness', style = {**word_style('1.5', 'left'), 'marginBottom': '0.5vw'}),
                sw_dd,
                html.Div('Base', style = {**word_style('1.5', 'left'), 'marginBottom': '0.5vw'}),
                base_dd,
                html.Div('Topping', style = {**word_style('1.5', 'left'), 'marginBottom': '0.5vw'}),
                tp_dd], style = {'width': '16vw', **put_vertical('left'), **add_border()}
            ),
            # drink and name
            html.Div(children = [
                pic('cup'),
                html.Div('', id = 'name_text', style = {'height': '2vw', **word_style('1', 'center'), 'width': '85%'})
                ], style = {'width': '15vw', **put_vertical('center'), **add_border(), 'marginLeft': '1vw'}
            ),
            # price and calories
            html.Div(children = [
                html.Div('Price', style = {**word_style('2', 'center'), 'marginBottom': '1vw'}),
                html.Div('--', id = 'price_text', style = {**word_style('5', 'center'), 'marginBottom': '3vw'}),
                html.Div('Calories', style = {**word_style('2', 'center'), 'marginBottom': '1vw'}),
                html.Div('--', id = 'cal_text', style = word_style('5', 'center'))
                ], style = {'width': '12vw', **put_vertical('center'), **add_border()}
            ),
            # scatter plot
            html.Div(scatter_plot, style = {'width': '37vw', 'height': '100%', **add_border()})
        
            # dashboard 4
        ],
        style = {**put_horizontal('center'), **block_style('full'), 'padding': '0vw 0vw 0vw 5vw', 'gap': '0.5vw'}
    )

# # 設定網頁的外觀 (Layout)
# app.layout = html.Div([
    

# # 修改最外層背景色
# ], style={'backgroundColor': '#EBDEC1', 'minHeight': '100vh', 'padding': '20px'})


# dashboard 1 callback
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

# dashboard 2 callback
@app.callback(  
    Output('output', 'children'),  
    Input('ingredient', 'value')  
)
def show_drinks(ingr_id):
    
    conn = sqlite3.connect('Tea_Shop_Database.db')
    
    # 情況1：沒有選擇任何成分
    if not ingr_id:
        query_all = """
        SELECT 
            d.drink_name,
            b.base_price,
            t.tp_price,
            sw.sw_cal,
            t.tp_cal,
            t.tp_caff
        FROM Drink d
        JOIN Base b ON d.base_id = b.base_id
        LEFT JOIN DrinkTopping dt ON d.drink_id = dt.drink_id
        LEFT JOIN Topping t ON dt.tp_id = t.tp_id
        LEFT JOIN DrinkSweetness ds ON d.drink_id = ds.drink_id
        LEFT JOIN Sweetness sw ON ds.sw_id = sw.sw_id
        WHERE sw.sw_name = 'Less Sugar'
        """
        
        # 單獨算 Base
        query_base_all = """
        SELECT 
            d.drink_id,
            d.drink_name,
            SUM(i.ingr_cal * bi.ingr_ml / 100.0) as base_cal,
            SUM(i.ingr_caff * bi.ingr_ml / 100.0) as base_caff
        FROM Drink d
        JOIN Base b ON d.base_id = b.base_id
        JOIN BaseIngredient bi ON b.base_id = bi.base_id
        JOIN Ingredient i ON bi.ingr_id = i.ingr_id
        GROUP BY d.drink_id
        """
        
        df = pd.read_sql_query(query_all, conn)
        df_base = pd.read_sql_query(query_base_all, conn)
    
    # 情況2：選擇特定成分
    else:
        query = """
        SELECT 
            d.drink_name,
            b.base_price,
            t.tp_price,
            sw.sw_cal,
            t.tp_cal,
            t.tp_caff
        FROM Drink d
        JOIN Base b ON d.base_id = b.base_id
        JOIN BaseIngredient bi ON b.base_id = bi.base_id
        LEFT JOIN DrinkTopping dt ON d.drink_id = dt.drink_id
        LEFT JOIN Topping t ON dt.tp_id = t.tp_id
        LEFT JOIN DrinkSweetness ds ON d.drink_id = ds.drink_id
        LEFT JOIN Sweetness sw ON ds.sw_id = sw.sw_id
        WHERE bi.ingr_id = ? AND sw.sw_name = 'Less Sugar'
        """
        
        # 單獨算 Base
        query_base = """
        SELECT 
            d.drink_id,
            d.drink_name,
            SUM(i.ingr_cal * bi.ingr_ml / 100.0) as base_cal,
            SUM(i.ingr_caff * bi.ingr_ml / 100.0) as base_caff
        FROM Drink d
        JOIN Base b ON d.base_id = b.base_id
        JOIN BaseIngredient bi ON b.base_id = bi.base_id
        JOIN Ingredient i ON bi.ingr_id = i.ingr_id
        WHERE d.drink_id IN (
            SELECT DISTINCT d2.drink_id
            FROM Drink d2
            JOIN Base b2 ON d2.base_id = b2.base_id
            JOIN BaseIngredient bi2 ON b2.base_id = bi2.base_id
            WHERE bi2.ingr_id = ?
        )
        GROUP BY d.drink_id
        """
        
        df = pd.read_sql_query(query, conn, params=(ingr_id,))
        df_base = pd.read_sql_query(query_base, conn, params=(ingr_id,))
    
    conn.close()
    
    # 資料處理
    df['tp_price'] = df['tp_price'].fillna(0)
    df['sw_cal'] = df['sw_cal'].fillna(0)
    df['tp_cal'] = df['tp_cal'].fillna(0)
    df['tp_caff'] = df['tp_caff'].fillna(0)
    
    # 合併 Base 營養資訊
    df = df.merge(df_base[['drink_name', 'base_cal', 'base_caff']], on='drink_name', how='left')
    df['base_cal'] = df['base_cal'].fillna(0)
    df['base_caff'] = df['base_caff'].fillna(0)
    
    # 計算總和
    df['Price'] = (df['base_price'] + df['tp_price']).astype(int)
    df['Calories'] = (df['base_cal'] + df['sw_cal'] + df['tp_cal']).astype(int)
    df['Caffeine'] = (df['base_caff'] + df['tp_caff']).astype(int)
    
    result = df[['drink_name', 'Price', 'Calories', 'Caffeine']].copy()
    result.columns = ['Name', 'Price', 'Kcal', 'Caffeine(mg)']
    result = result.drop_duplicates()
    
    # 表格
    return html.Div([
        dash_table.DataTable(
            data=result.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in result.columns],
            sort_action='native', # 排序功能
            page_action='native', # 分頁功能
            page_size=10, # 10筆一頁
            
            # 表頭
            style_header={
                'backgroundColor': '#8B6F47', 
                'color': 'white', 
                'fontWeight': 'bold',
                'fontSize': '14px',
                'textAlign': 'center',
                'padding': '12px',
                'border': 'none',  
            },
            
            # 儲存格
            style_cell={
                'textAlign': 'center',
                'padding': '12px 10px',
                'backgroundColor': 'white',
                'border': 'none',
                'borderBottom': '1px solid #E7DABD',
                'fontSize': '14px',
                'color': '#333'
            },
            
            # Name 欄靠左
            style_cell_conditional=[
                {
                    'if': {'column_id': 'Name'},
                    'textAlign': 'left',
                    'fontWeight': '500',
                    'paddingLeft': '15px'
                }
            ],
            
            # 偶數變色
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': "#F4F0E9"
                }
            ],
        ),
        
        html.Div(
            '* Calories are calculated based on Less Sugar level.',
            style={
                'marginTop': '20px',
                'fontStyle': 'italic',
                'color': '#999',
                'fontSize': '13px',
                'textAlign': 'center'
            }
        )
    ])

# dashboard 3 callback
# click dropdown -> change plot, picture, drink name, price, calories, base options, tp options
@callback(
  Output('scatter_plot', 'figure'),
  Output('drink_pic', 'src'),
  Output('price_text', 'children'),
  Output('cal_text', 'children'),
  Output('name_text', 'children'),
  Output('base_dd', 'options'),
  Output('tp_dd', 'options'),
  Output('base_options_store','data'),  # store
  Output('tp_options_store','data'),  # store
  Input('sw_dd', 'value'),
  Input('base_dd', 'value'),
  Input('tp_dd', 'value'),
  Input('scatter_plot', 'clickData'),
  State('base_options_store','data'),  # store
  State('tp_options_store','data')  # store
)
def update_scatter(sw, base, tp, clickData, base_options_data, tp_options_data):
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    new_fig = scatter_fig  # default
    img = '/assets/cup.png'
    price = '--'
    cal = '--'
    name = ''
    base_options = base_options_data.copy()
    tp_options = tp_options_data.copy()  

    # change data and plot
    new_drink_info = drink_info.copy()
    if sw is not None:
        new_drink_info = new_drink_info.loc[new_drink_info['drink_full_name'].str.contains(sw)]
    if base is not None:
        new_drink_info = new_drink_info.loc[new_drink_info['drink_base'] == base]
    if tp is not None:
        drinks_with_tp = df_tp.loc[df_tp['tp_name'] == tp, 'drink_name'].unique()
        new_drink_info = new_drink_info[new_drink_info['drink_name'].isin(drinks_with_tp)]

    new_fig = px.scatter(
        data_frame = new_drink_info,
        x = 'drink_cal',
        y = 'drink_price',
        color = 'drink_name',
        hover_data = ['drink_full_name'],
        custom_data = ['drink_full_name'],
        range_x = [-10, 810],
        range_y = [28, 82]
    )
    new_fig.update_layout(
        font = dict(family = 'Arial, sans-serif', size = 16, color = COLOR_DARK_BROWN),
        plot_bgcolor = COLOR_WHITE,
        xaxis = plot_grid('Calories'),
        yaxis = plot_grid('Price')
    )

    # change pic, name, price, cal, options
    if trigger_id == 'sw_dd' and sw is not None:  # change plot
        if len(new_drink_info['drink_name'].unique().tolist()) == 1:
            img = to_pic(new_drink_info['drink_name'].unique().tolist()[0])
        elif base is not None:
            img = to_pic(base)
        elif tp != 'NULL':
            img = to_pic(tp)

    elif trigger_id == 'base_dd' and base is not None:  # change data (plot), pic, topping
        if len(new_drink_info['drink_name'].unique().tolist()) == 1:
            img = to_pic(new_drink_info['drink_name'].unique().tolist()[0])
        else:
            img = to_pic(base)

        tp_options = []
        for idx, row in drink_info.iterrows():
            current_tp = find_tp(row['drink_full_name'])
            current_base = row['drink_base']
            if current_base == base and current_tp not in tp_options:
                tp_options.append(current_tp)

    elif trigger_id == 'tp_dd' and tp is not None:  # change data (plot), pic, base
        if len(new_drink_info['drink_name'].unique().tolist()) == 1:
            img = to_pic(new_drink_info['drink_name'].unique().tolist()[0])
        elif tp != 'NULL':
            img = to_pic(tp)

        base_options = []
        for idx, row in drink_info.iterrows():
            current_tp = find_tp(row['drink_full_name'])
            current_base = row['drink_base']
            if current_tp == tp and current_base not in base_options:
                base_options.append(current_base)
        
    elif trigger_id == 'scatter_plot' and clickData is not None:  # change price, name, cal, img
        name = clickData['points'][0]['customdata'][0]
        price = drink_info.loc[drink_info['drink_full_name'] == name, 'drink_price']
        cal = drink_info.loc[drink_info['drink_full_name'] == name, 'drink_cal']
        img = to_pic(name)
    elif base is None and tp is None:  # clear base and tp
        base_options = pd.Series(base_list).unique().tolist()
        tp_options = tp_list
    return new_fig, img, price, cal, name, base_options, tp_options, base_options, tp_options

if __name__ == '__main__':
    app.run(debug=True)

# import dash
# from dash import dcc, html, dash_table, Input, Output
# import pandas as pd
# import sqlite3

# # 連接資料庫
# DB = 'Tea_Shop_Database.db'
# conn = sqlite3.connect(DB)  
# conn.close() 

# # Dash 
# app = dash.Dash(__name__)  

# app.layout = html.Div([
#     # 外層（米色背景）
#     html.Div([
        

#     ], style={
#         'backgroundColor': "#EBDEC1",
#         'minHeight': '100vh',
#         'padding': '40px 20px',
#     })
# ])

# # Callback


# if __name__ == '__main__':
#     app.run(debug=True)

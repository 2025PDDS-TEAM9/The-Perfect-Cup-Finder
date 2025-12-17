import sqlite3
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback
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
    # print(drink)
    file = drink.split(' (')[0].replace("'", '').replace(' ', '_').lower()
    # print(file)
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
# base_list = df_ingr['base_name'].unique().tolist()
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

# print(type(df_ingr['base_name'].unique().tolist()))
# print(df_ingr['base_name'].unique())

app = Dash(__name__, prevent_initial_callbacks = True)

app.layout = html.Div(
        children = [
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
        ],
        style = {**put_horizontal('center'), **block_style('full'), 'padding': '0vw 0vw 0vw 5vw', 'gap': '0.5vw'}
    )

# click dropdown -> change plot, picture, drink name, price, calories, base options, tp options
@callback(
  Output('scatter_plot', 'figure'),
  Output('drink_pic', 'src'),
  Output('price_text', 'children'),
  Output('cal_text', 'children'),
  Output('name_text', 'children'),
  Output('base_dd', 'options'),
  Output('tp_dd', 'options'),
  Input('sw_dd', 'value'),
  Input('base_dd', 'value'),
  Input('tp_dd', 'value'),
  Input('scatter_plot', 'clickData'),
  allow_duplicate = True
)
def update_scatter(sw, base, tp, clickData):
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    new_fig = scatter_fig  # default
    img = '/assets/cup.png'
    price = '--'
    cal = '--'
    name = ''
    base_options = pd.Series(base_list).unique().tolist()
    tp_options = tp_list
    # if sw is None and base is None and tp is None and clickData is None:
        

    # change data and plot
    new_drink_info = drink_info.copy()
    if sw is not None:
        new_drink_info = new_drink_info.loc[new_drink_info['drink_full_name'].str.contains(sw)]
    if base is not None:
        new_drink_info = new_drink_info.loc[new_drink_info['drink_base'] == base]
    if tp is not None:
        # new_drink_info = new_drink_info[new_drink_info['drink_name'].str.contains(tp)]
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
    # print(new_drink_info.head())
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
        for idx, row in new_drink_info.iterrows():
            current_tp = find_tp(row['drink_full_name'])
            if current_tp not in tp_options:
                tp_options.append(current_tp)

    elif trigger_id == 'tp_dd' and tp is not None:  # change data (plot), pic, base
        if len(new_drink_info['drink_name'].unique().tolist()) == 1:
            img = to_pic(new_drink_info['drink_name'].unique().tolist()[0])
        elif tp != 'NULL':
            img = to_pic(tp)
        base_options = []
        for item in new_drink_info.loc[: , 'drink_base']:
            if item not in base_options:
                base_options.append(item)
        
    elif trigger_id == 'scatter_plot' and clickData is not None:  # change price, name, cal, img
        name = clickData['points'][0]['customdata'][0]
        price = drink_info.loc[drink_info['drink_full_name'] == name, 'drink_price']  # .iloc[0]
        cal = drink_info.loc[drink_info['drink_full_name'] == name, 'drink_cal']  # .iloc[0]
        img = to_pic(name)
    return new_fig, img, price, cal, name, base_options, tp_options

if __name__ == '__main__':
	app.run(debug = True)
     
# note
# 飲料的ingr: 
# (drink_id), drink_name, (base_id), base_name, base_price, (ingr_id), ingr_ml, ingr_name, ingr_cal, ingr_caff

# 飲料的topping:
# (drink_id), drink_name, (tp_id), tp_name, tp_price, tp_cal, tp_caff

# 飲料的甜度:
# (drink_id), drink_name, (sw_id), sw_name, sw_cal
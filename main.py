import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
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

cursor.execute("SELECT drink_name FROM Drink")
data_drink = cursor.fetchall()


conn.close()

# style definition
COLOR_LIGHT_BROWN = '#EBDEC1'
COLOR_MIDDLE_BROWN = '#F4F0E9'
COLOR_DARK_BROWN = '#5C4033'
COLOR_WHITE = '#FFFFFF'
COLOR_BLACK = '#000000'
FONT_FAMILY = 'Arial, sans-serif'
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
    if mode == 'full':
        return {'backgroundColor': COLOR_WHITE,
                'borderRadius': '20px',
                'margin': '1vw 5vw 5vw 5vw',
                'height': '80vh',
                'width': '80vw',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'}
    else:  # half
        return {'backgroundColor': COLOR_WHITE,
                'borderRadius': '20px',
                'margin': '1vw 2vw 5vw 2vw',
                'height': '80vh',
                'width': '40vw',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'}
    

def word_style(size, alignment):
    return {'color': COLOR_DARK_BROWN,
            'fontFamily': FONT_FAMILY,
            'fontSize': f'{size}vw',
            'fontWeight': 'normal',  # bold, normal, 100~900
            'textAlign': alignment}

def dd_style():
    return {'color': COLOR_BLACK,
            'fontFamily': FONT_FAMILY,
            'fontSize': '1vw',
            'fontWeight': 'normal',  # bold, normal, 100~900
            'textAlign': 'left',
            'borderRadius': '4px',
            }

def dd_border(ratio):
    return {'border': '1px solid #5C4033',
            'borderRadius': '5px',
            'width': ratio}

def put_vertical(alignment):
    return {'display': 'flex', 'flexDirection': 'column', 'alignItems': alignment, 'justifyContent': 'center'}

def put_horizontal(alignment):
    return {'display': 'flex', 'flexDirection': 'row', 'alignItems': alignment, 'justifyContent': 'center'}

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

def title_style(size, weight, marginTop, marginBottom):
    return {'color': COLOR_DARK_BROWN,
            'fontFamily': FONT_FAMILY,
            'fontSize': f'{size}vw',
            'fontWeight': weight,  # bold, normal, 100~900
            'textAlign': 'center',
            'marginTop': f'{marginTop}vw',
            'marginBottom': f'{marginBottom}vw'}

def title(title1, title2, type):
    if type == 'big':
        return html.Div([
                    html.Div(title1, style = title_style(3, 900, 2, 0.5)),
                    html.Div(title2, style = title_style(2.5, 700, 0, 4))
                ], style = put_vertical('center'))
    else:
        return html.Div([
                    html.Div(title1, style = title_style(2, 750, 0, 0.5)),
                    html.Div(title2, style = title_style(1.7, 500, 0, 0)),
                ], style = put_vertical('center'))

def default_sw(text):
    return html.Div(
                text,
                style={
                    'fontStyle': 'italic',
                    'color': '#999',
                    'fontSize': '13px',
                    'textAlign': 'center',
                    'fontFamily': FONT_FAMILY,
                    **add_border()
                }
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

# base info per drink (base_name/base_price)
base_info = (
    df_ingr.groupby("drink_name", as_index=False)
    .agg(base_name=("base_name", "first"), base_price=("base_price", "first"))
)

# base calories/caffeine from ingredients
base_cal = (
    df_ingr.assign(base_cal_part=lambda d: d["ingr_cal"] * d["ingr_ml"] / 100)
    .groupby("drink_name", as_index=False)["base_cal_part"]
    .sum()
    .rename(columns={"base_cal_part": "base_cal"})
)

base_caff = (
    df_ingr.assign(base_caff_part=lambda d: d["ingr_caff"] * d["ingr_ml"] / 100)
    .groupby("drink_name", as_index=False)["base_caff_part"]
    .sum()
    .rename(columns={"base_caff_part": "base_caff"})
)

# topping aggregation per drink (support multiple toppings)
tp_agg = (
    df_tp[df_tp["tp_name"] != "NULL"]
    .groupby("drink_name", as_index=False)
    .agg(
        tp_price=("tp_price", "sum"),
        tp_cal=("tp_cal", "sum"),
        tp_caff=("tp_caff", "sum"),
        toppings=("tp_name", "count"),
    )
)

# drinks with no topping -> fill 0
drink_base = pd.DataFrame({"drink_name": drink_list})
drink_base = drink_base.merge(base_info, on="drink_name", how="left")
drink_base = drink_base.merge(base_cal, on="drink_name", how="left")
drink_base = drink_base.merge(base_caff, on="drink_name", how="left")
drink_base = drink_base.merge(tp_agg, on="drink_name", how="left")

for c in ["base_price", "base_cal", "base_caff", "tp_price", "tp_cal", "tp_caff", "toppings"]:
    if c in drink_base.columns:
        drink_base[c] = drink_base[c].fillna(0)

# pick default sweetness per drink: Less Sugar if exists else first
def pick_default_sweetness(df, prefer="Less Sugar"):
    rows = []
    for dn, g in df.groupby("drink_name"):
        prefer_rows = g[g["sw_name"] == prefer]
        rows.append(prefer_rows.iloc[0] if not prefer_rows.empty else g.iloc[0])
    return pd.DataFrame(rows)

df_sw_default = pick_default_sweetness(df_sw, prefer="Less Sugar")

# final metrics table (default sweetness)
drink_info_2 = drink_base.merge(df_sw_default, on="drink_name", how="left")

# Price = base_price + Σ(tp_price)
drink_info_2["price"] = drink_info_2["base_price"] + drink_info_2["tp_price"]

# Calories = Σ(ingr_cal*ml/100) + Σ(tp_cal) + sw_cal
drink_info_2["calories"] = drink_info_2["base_cal"] + drink_info_2["tp_cal"] + drink_info_2["sw_cal"]

# Caffeine = Σ(ingr_caff*ml/100) + Σ(tp_caff)
drink_info_2["caffeine"] = drink_info_2["base_caff"] + drink_info_2["tp_caff"]

# Keep only needed columns
plot_df = drink_info_2[["drink_name", "price", "calories", "caffeine", "toppings", "sw_name"]].copy()

# dashboard elements
# dashboard 1
bar_dd = dcc.Dropdown(
                    id='metric-dropdown',
                    options=[
                        {'label': 'Calories', 'value': 'drink_cal'},
                        {'label': 'Price', 'value': 'drink_price'},
                        {'label': 'Caffeine', 'value': 'drink_caff'}
                    ],
                    value='drink_cal',
                    clearable=False,
                    style=dd_style()
                )
bar_radiobtn = dcc.RadioItems(
                    id='sort-order',
                    options=[
                        {'label': 'From High to Low', 'value': 'descending'},
                        {'label': 'From Low to High', 'value': 'ascending'}
                    ],
                    value='descending',
                    style = {**dd_style(), **put_vertical('center')}
                    # style={'marginLeft': '10px', 'display': 'inline-block', 'color': '#5C4033'} # 選項文字也改深色
                )
bar_select = html.Div([
                # sort by
                html.Div([
                    html.Label("Sort by", style=word_style('1', 'left')), # 標籤也改深色
                    html.Div(bar_dd, style = dd_border('60%'))
                ], style = {**put_horizontal('center'), 'width': '16vw', **add_border(), 'gap': '1vw'}),
                # order
                html.Div([
                    html.Label("Order", style=word_style('1', 'left')), # 標籤也改深色
                    bar_radiobtn
                ], style = {**put_horizontal('center'), 'width': '16vw', **add_border(), 'gap': '1vw'})  
            ], style={**put_horizontal('center'), 'width': '35vw', 'marginTop': '4vh', 'marginBottom': '3vh'})
# dashboard 2
table_dd = html.Div([
                html.Div('Select Ingredient:', 
                    style={'fontWeight': '500', 'display': 'block', **word_style('1.2', 'left')}),
                html.Div(
                    dcc.Dropdown(
                        id='ingredient',
                        options=[{'label': row['ingr_name'], 'value': row['ingr_id']} 
                            for _, row in ingredients.iterrows()],
                        placeholder='Select a Ingredients',
                        style={**dd_style()}
                ), style = dd_border('50%')),
            ], style = {**put_horizontal('center'), **add_border(), 'marginTop': '4vh', 'marginBottom': '3vh', 'width': '30vw', 'gap': '2vh'})
table = html.Div([            
                html.Div(id='output')
            ], style={
                **add_border(),
                'height': '54vh',
                'width': '35vw'
            })

# dashboard 3
sw_dd = dcc.Dropdown(
            id = 'sw_dd',
            options = [
                {'label': 'Full Sugar (100%)', 'value': 'Full Sugar'},
                {'label': 'Half Sugar (50%)', 'value': 'Half Sugar'},
                {'label': 'Less Sugar (20%)', 'value': 'Less Sugar'},
                {'label': 'No Sugar (0%)', 'value': 'No Sugar'}
            ],
            placeholder = 'Select a Sweetness',
            style = {**dd_style()}
        )
base_dd = dcc.Dropdown(id = 'base_dd', options = base_list, 
                       placeholder = 'Select a Base', style = {**dd_style()})
tp_dd = dcc.Dropdown(id = 'tp_dd', options = tp_list,
                     placeholder = 'Select a Topping', style = dd_style())
scatter_fig = px.scatter(
  data_frame = drink_info,
  x = 'drink_cal',
  y = 'drink_price',
  color = 'drink_name',
  hover_data = ['drink_full_name'],
  custom_data = ['drink_full_name'],
  range_x = [0, 850],
  range_y = [20, 85]
)
scatter_fig.update_layout(
    font = dict(family = 'Arial, sans-serif', size = 14, color = COLOR_DARK_BROWN),
    plot_bgcolor = COLOR_WHITE,
    xaxis = plot_grid('Calories (Kcal)'),
        yaxis = plot_grid('Price (NTD)'),
    showlegend = False,
    margin = dict(l = 0, r = 0, t = 0, b = 0)
)
scatter_plot = dcc.Graph(
		    id = 'scatter_plot',
		    figure = scatter_fig,
		    style = {'width': '100%', 'height': '100%'}
            )
scatter_dd = html.Div(children = [
                html.Div([
                    html.Div('Sweetness', style = {**word_style('1', 'left'), 'marginBottom': '0.5vw'}),
                    html.Div(sw_dd, style = dd_border('100%'))
                ]),
                html.Div([
                    html.Div('Base', style = {**word_style('1', 'left'), 'marginBottom': '0.5vw'}),
                    html.Div(base_dd, style = dd_border('100%'))
                ]),
                html.Div([
                    html.Div('Topping', style = {**word_style('1', 'left'), 'marginBottom': '0.5vw'}),
                    html.Div(tp_dd, style = dd_border('100%'))
                ])], style = {'width': '12vw', **put_vertical('left'), **add_border(), 'gap': '1vw'}
            )
scatter_cup = html.Div(children = [
                pic('cup'),
                html.Div('', id = 'name_text', style = {'height': '2vw', **word_style('1', 'center'), 'width': '85%'})
                ], style = {'width': '12vw', **put_vertical('center'), **add_border(), 'marginLeft': '1vw'}
            )
scatter_big_text = html.Div(children = [
                html.Div('Price', style = {**word_style('1.5', 'center'), 'marginBottom': '1vw'}),
                html.Div('--', id = 'price_text', style = {**word_style('3', 'center'), 'marginBottom': '3vw'}),
                html.Div('Calories', style = {**word_style('1.5', 'center'), 'marginBottom': '1vw'}),
                html.Div('--', id = 'cal_text', style = word_style('3', 'center'))
                ], style = {'width': '12vw', **put_vertical('center'), **add_border()}
            )

# dashboard 4
# 4) Radar chart (4 metrics) + normalize 0–10
RADAR_COLS = {
    "Calories": "calories",
    "Price": "price",
    "Caffeine": "caffeine",
    "Toppings": "toppings",
}

# strong scaling for Calories & Price: percentile(rank) on log1p values
def transform(metric, x):
    x = float(x)
    if metric in ("Calories", "Price"):
        return float(np.log1p(max(x, 0)))
    return x

# precompute sorted distributions (after transform) for percentile scaling
dist = {}
bounds = {}
for metric, col in RADAR_COLS.items():
    s = plot_df[col].astype(float).apply(lambda v: transform(metric, v))
    dist[metric] = np.sort(s.to_numpy())

    lo = float(np.quantile(dist[metric], 0.05))
    hi = float(np.quantile(dist[metric], 0.95))
    if hi <= lo:  # fallback
        lo = float(dist[metric].min())
        hi = float(dist[metric].max())
    bounds[metric] = (lo, hi)

def percentile_0_10(metric, val):
    arr = dist[metric]
    if arr.size <= 1:
        return 0.0
    x = transform(metric, val)
    # clamp to robust bounds before percentile
    lo, hi = bounds[metric]
    x = max(lo, min(x, hi))
    # percentile rank (0~1)
    p = np.searchsorted(arr, x, side="left") / (arr.size - 1)
    return 10.0 * float(p)

def norm_0_10(metric, val):
    if metric in ("Calories", "Price"):
        return percentile_0_10(metric, val)

    # 其他指標用 robust min-max
    lo, hi = bounds[metric]
    x = transform(metric, val)
    x = max(lo, min(x, hi))
    if hi - lo == 0:
        return 0.0
    return 10.0 * (x - lo) / (hi - lo)


def radar_fig(left_drink, right_drink):
    cats = list(RADAR_COLS.keys())
    cats_closed = cats + [cats[0]]

    def get_row(drink):
        if not drink:
            return None
        row = plot_df.loc[plot_df["drink_name"] == drink]
        return None if row.empty else row.iloc[0]

    def rvals(drink):
        row = get_row(drink)
        if row is None:
            vals = [0.0] * len(cats)
        else:
            vals = [norm_0_10(c, row[RADAR_COLS[c]]) for c in cats]
        return vals + [vals[0]]

    def raw_vals_for_hover(drink):
        """依照 theta 順序回傳『該點對應 metric 的原始值』；最後一個重複首點"""
        row = get_row(drink)
        if row is None:
            raw = [None] * len(cats)
        else:
            raw = [float(row[RADAR_COLS[c]]) for c in cats]
        raw_closed = raw + [raw[0]]
        return raw_closed

    HOVER_TMPL_LEFT = "<b>Left</b><br>Metric: %{theta}<br>Value: %{customdata}<extra></extra>"
    HOVER_TMPL_RIGHT = "<b>Right</b><br>Metric: %{theta}<br>Value: %{customdata}<extra></extra>"

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=rvals(left_drink),
        theta=cats_closed,
        fill="toself",
        name="Left",
        customdata=raw_vals_for_hover(left_drink),
        hovertemplate=HOVER_TMPL_LEFT,
    ))

    fig.add_trace(go.Scatterpolar(
        r=rvals(right_drink),
        theta=cats_closed,
        fill="toself",
        name="Right",
        customdata=raw_vals_for_hover(right_drink),
        hovertemplate=HOVER_TMPL_RIGHT,
    ))

    fig.update_layout(
        paper_bgcolor=COLOR_WHITE,
        plot_bgcolor=COLOR_WHITE,
        font=dict(family=FONT_FAMILY, size=14, color=COLOR_DARK_BROWN),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        polar=dict(
            domain=dict(x=[0.07, 0.93], y=[0.07, 0.93]),
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                showticklabels=False,
                gridcolor=COLOR_LIGHT_BROWN,
            ),
            angularaxis=dict(
                gridcolor=COLOR_LIGHT_BROWN,
                tickfont=dict(size=14),
                rotation=90,
                direction="clockwise",
            ),
        ),
    )
    return fig

drink_options = [{"label": n, "value": n} for n in sorted(plot_df["drink_name"].unique())]

radar_left_dd = dcc.Dropdown(
                    id="left_dd",
                    options=drink_options,
                    placeholder="Select a Drink",
                    value=None,
                    style=dd_style(),
                )
radar_left_cup = html.Img(id="left_img", src="/assets/cup.png", style={"width": "70%"})
radar_left_name = html.Div(id="left_name", style={**word_style("1.0", "center"), "minHeight": "2vw"})

radar = dcc.Graph(
            id="radar",
            figure=radar_fig(None, None),
            style={'height': '100%', 'width': '100%', **add_border()}
        )

radar_right_dd = dcc.Dropdown(
                    id="right_dd",
                    options=drink_options,
                    placeholder="Select a Drink",
                    value=None,
                    style=dd_style(),
                )
radar_right_cup = html.Img(id="right_img", src="/assets/cup.png", style={"width": "70%"})
radar_right_name = html.Div(id="right_name", style={**word_style("1.0", "center"), "minHeight": "2vw"})

app = Dash()

app.layout = html.Div(children = 
                html.Div([
                    title('The Perfect Cup Finder', 'Your Drink Decision Guide', 'big'),

                    # dashboard 1 and 2 (horizontal)
                    html.Div([
                        # dashboard 1
                        html.Div([
                            title('Start Your Drink Journey:', "What's Trending on the Menu?", 'small'),
                            html.Div([
                                bar_select,
                                html.Div(dcc.Graph(id='bar-chart'), style = {'width': '35vw', 'height': '56vh', **add_border()}),
                                html.Div(default_sw('* Default sweetness for all drinks: No Sugar.'),
                                         style = {'marginBottom': '5vh', 'marginTop': '2vh'})
                            ], style = {**block_style('half'), **put_vertical('center')})
                        ], style = put_vertical('center')),
                        # dashboard 2
                        html.Div([
                            title('Choose Your Flavor Ingredient:', 'Discover Drinks Made Just for You', 'small'),
                            html.Div([
                                table_dd,
                                html.Div([
                                    table,
                                    html.Div(default_sw('* Calories are calculated based on Less Sugar level.'),
                                        style = {'marginBottom': '5vh', 'marginTop': '2vh'})
                                ], style = {**put_vertical('center')})
                            ], style = {**block_style('half'), **put_vertical('left')})
                        ], style = put_vertical('center'))
                    ], style = put_horizontal('center')),
                    # dashboard 3
                    html.Div([
                        title('Build Your Perfect Cup:', 'Customize Sweetness, Base, and Toppings', 'small'),
                        html.Div([
                            # memory for base and topping options
                            dcc.Store(id = 'base_options_store', data = pd.Series(base_list).unique().tolist()),
                            dcc.Store(id = 'tp_options_store', data = tp_list),

                            scatter_dd, scatter_cup, scatter_big_text,
                            # scatter plot
                            html.Div(scatter_plot, style = {'width': '40vw', 'height': '100%', **add_border()})
                        ], style = {**put_horizontal('center'), **block_style('full'), 'padding': '0vw 0vw 0vw 5vw', 'gap': '0.5vw'})
                    ], style = put_vertical('center')),
                    # dashboard4
                    html.Div([
                        title('Final Decision:', 'Compare Your Top Picks Before You Sip!', 'small'),
                        html.Div([
                            # dd and plot
                            html.Div([
                                # left
                                html.Div([
                                    html.Div(radar_left_dd, style = dd_border('100%')), radar_left_cup, radar_left_name
                                ], style = {**add_border(), **put_vertical('center'), 'width': '15vw', 'gap': '4vh'}),
                                # radar
                                html.Div(radar, style = {'height': '60vh', 'width': '80vh'}),
                                # right
                                html.Div([
                                    html.Div(radar_right_dd, style = dd_border('100%')), radar_right_cup, radar_right_name
                                ], style = {**add_border(), **put_vertical('center'), 'width': '15vw', 'gap': '4vh'})
                            ], style = {**put_horizontal('center'), 'gap': '1vw'}),
                            # annotation
                            default_sw('* Default sweetness for all drinks: Less Sugar (fallback to first available if missing).')
                        ], style = {**put_vertical('center'), **block_style('full'), 'gap': '4vh'})
                    ], style = put_vertical('center'))
                ], style = put_vertical('center')))

# dashboard 1 callback
@callback(
    Output('bar-chart', 'figure'),
    Input('metric-dropdown', 'value'),
    Input('sort-order', 'value')
)
def update_graph(selected_metric, sort_order):
    # 根據選擇的項目排序
    is_ascending = True if sort_order == 'ascending' else False
    
    sorted_df = drink_info_no_sugar.sort_values(by=selected_metric, ascending=is_ascending)
    
    # 取出前 10 筆資料
    top_10_df = sorted_df.head(10)
    

    labels_map = {
        'drink_cal': 'Calories (Kcal)',
        'drink_price': 'Price (NTD)',
        'drink_caff': 'Caffeine (mg)',
        'drink_name': 'Drink'
    }

    # 3. 畫圖
    fig = px.bar(
        top_10_df,
        x='drink_name',         
        y=selected_metric,      
        text=selected_metric,   
        labels=labels_map
    )
    
    
    fig.update_traces(textposition='inside',
                      marker_color=COLOR_DARK_BROWN,
                      textfont = dict(size = 6)) # 我順便把柱子的顏色也改成深咖啡色，看看你喜不喜歡

    fig.update_xaxes(
        tickmode = 'array',
        tickvals = top_10_df['drink_name'],
        title_text = None,
        tickangle = 45,
        tickfont = dict(size = 8)
        # ticktext = [name.replace(' ', '<br>') for name in top_10_df['drink_name']]
    )
    fig.update_yaxes(tickfont = dict(size = 10))

    fig.update_layout(
        xaxis={'categoryorder':'total ascending'} if is_ascending else {'categoryorder':'total descending'},
        
        # --- 顏色的設定 ---
        plot_bgcolor='white',      
        font_color=COLOR_DARK_BROWN,       
        title_font_color=COLOR_DARK_BROWN,
        # -------------------------
        title_x = 0.5,
        autosize = True,
        margin = dict(l = 0, r = 0, t = 0, b = 0)
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
    result.columns = ['Name', 'Price (NTD)', 'Calories (Kcal)', 'Caffeine (mg)']
    columns = [
        {'name': ['Name'], 'id': 'Name'},
        {'name': ['Price', 'NTD'], 'id': 'Price'},
        {'name': ['Calories', 'Kcal'], 'id': 'Calories'},
        {'name': ['Caffeine', 'mg'], 'id': 'Caffeine'},
    ]
    result = result.drop_duplicates()
    
    # 表格
    return html.Div([
        dash_table.DataTable(
            data=result.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in result.columns],
            sort_action='native', # 排序功能
            page_action='native', # 分頁功能
            page_size=7, # 7筆一頁
            style_table = {
                'height': '100%',
                'overflowY': 'auto',
                'width': '100%',
                'margin': '0',
                'padding': '0',
                'borderSpacing': '0'
            },
            
            # 表頭
            style_header={
                'backgroundColor': '#8B6F47', 
                'color': 'white', 
                'fontWeight': 'bold',
                'fontSize': '12px',
                'textAlign': 'center',
                'padding': '6px',
                'border': 'none',  
                'height': 'auto'
            },
            
            # 儲存格
            style_cell={
                'textAlign': 'center',
                'padding': '6px 5px',
                'backgroundColor': 'white',
                'border': 'none',
                'borderBottom': '1px solid #E7DABD',
                'fontSize': '12px',
                'color': '#333',
                'whiteSpace': 'normal',   # 允許換行
                'height': 'auto'          # 高度自動調整
            },
            
            # Name 欄靠左
            style_cell_conditional=[
                {
                    'if': {'column_id': 'Name'},
                    'textAlign': 'left',
                    'fontWeight': '500',
                    'paddingLeft': '10px'
                }
            ],
            
            # 偶數變色
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': COLOR_MIDDLE_BROWN
                }
            ],
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
  Input('scatter_plot', 'hoverData'),
  State('base_options_store','data'),  # store
  State('tp_options_store','data')  # store
)
def update_scatter(sw, base, tp, hoverData, base_options_data, tp_options_data):
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    new_fig = scatter_fig  # default
    img = '/assets/cup.png'
    price = '--'
    cal = '--'
    name = ''
    base_options = base_options_data.copy()
    tp_options = tp_options_data.copy()  

    # filter all data
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
        font = dict(family = 'Arial, sans-serif', size = 14, color = COLOR_DARK_BROWN),
        plot_bgcolor = COLOR_WHITE,
        xaxis = plot_grid('Calories (Kcal)'),
        yaxis = plot_grid('Price (NTD)'),
        showlegend = False
    )

    # change options
    if trigger_id == 'base_dd':
        if tp is None:
            base_options = pd.Series(base_list).unique().tolist()
        if base is not None:
            tp_options = []
            for idx, row in drink_info.iterrows():
                current_tp = find_tp(row['drink_full_name'])
                current_base = row['drink_base']
                if current_base == base and current_tp not in tp_options:
                    tp_options.append(current_tp)
    elif trigger_id == 'tp_dd':
        if base is None:
            tp_options = tp_list
        if tp is not None:
            base_options = []
            for idx, row in drink_info.iterrows():
                current_tp = find_tp(row['drink_full_name'])
                current_base = row['drink_base']
                if current_tp == tp and current_base not in base_options:
                    base_options.append(current_base)

    # change pic, name, price, cal
    if len(new_drink_info['drink_name'].unique().tolist()) == 1:
        img = to_pic(new_drink_info.iloc[0]['drink_name'])
        if len(new_drink_info['drink_full_name'].unique().tolist()) == 1:
            price = new_drink_info.iloc[0]['drink_price']
            cal = new_drink_info.iloc[0]['drink_cal']
        
    

    elif trigger_id == 'sw_dd' and sw is not None:  # change plot
        if base is not None:
            img = to_pic(base)
        elif tp is not None and tp != 'NULL':
            img = to_pic(tp)
    elif trigger_id == 'sw_dd' and sw is None:
        price = '--'
        cal = '--'
    # name = ''
    elif trigger_id == 'base_dd' and base is not None:  # change data (plot), pic, topping
        img = to_pic(base)
    elif trigger_id == 'base_dd' and base is None:
        if tp is not None and tp != 'NULL':
            img = to_pic(tp)
    elif trigger_id == 'tp_dd' and tp is not None:  # change data (plot), pic, base
        if tp != 'NULL':
            img = to_pic(tp)
    elif trigger_id == 'tp_dd' and tp is None:
        if base is not None:
            img = to_pic(base)
        
    # change pic, name, price, cal through hover data points
    if trigger_id == 'scatter_plot' and hoverData is not None:  # change price, name, cal, img
        name = hoverData['points'][0]['customdata'][0]
        price = drink_info.loc[drink_info['drink_full_name'] == name, 'drink_price']
        cal = drink_info.loc[drink_info['drink_full_name'] == name, 'drink_cal']
        img = to_pic(name)

    return new_fig, img, price, cal, name, base_options, tp_options, base_options, tp_options

# dashboard 4 callback
@callback(
    Output("radar", "figure"),
    Output("left_img", "src"),
    Output("left_name", "children"),
    Output("right_img", "src"),
    Output("right_name", "children"),
    Input("left_dd", "value"),
    Input("right_dd", "value"),
)
def update_compare(left_drink, right_drink):
    fig = radar_fig(left_drink, right_drink)

    left_src = to_pic(left_drink) if left_drink else "/assets/cup.png"
    right_src = to_pic(right_drink) if right_drink else "/assets/cup.png"

    # 想顯示更清楚：drink_name + (default sweetness)
    def label(drink):
        if not drink:
            return ""
        sw = plot_df.loc[plot_df["drink_name"] == drink, "sw_name"].iloc[0]
        return f"{drink} ({sw})"

    return fig, left_src, label(left_drink), right_src, label(right_drink)

if __name__ == '__main__':
    app.run(debug=True)

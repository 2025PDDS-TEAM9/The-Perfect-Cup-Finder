import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, State, callback, dash_table, callback_context


# =========================
# 1) Load DB
# =========================
DB_PATH = "Tea_Shop_Database.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("SELECT drink_name FROM Drink")
data_drink = cursor.fetchall()

query_ingr = """
SELECT D.drink_name, B.base_name, B.base_price, BI.ingr_ml, I.ingr_name, I.ingr_cal, I.ingr_caff
FROM Drink AS D
JOIN Base AS B ON D.base_id = B.base_id
JOIN BaseIngredient AS BI ON B.base_id = BI.base_id
JOIN Ingredient AS I ON BI.ingr_id = I.ingr_id
"""
cursor.execute(query_ingr)
data_ingr = cursor.fetchall()

query_tp = """
SELECT D.drink_name, T.tp_name, T.tp_price, T.tp_cal, T.tp_caff
FROM Drink AS D
LEFT JOIN DrinkTopping AS DT ON D.drink_id = DT.drink_id
LEFT JOIN Topping AS T ON DT.tp_id = T.tp_id
"""
cursor.execute(query_tp)
data_tp = cursor.fetchall()

query_sw = """
SELECT D.drink_name, S.sw_name, S.sw_cal
FROM Drink AS D
LEFT JOIN DrinkSweetness AS DS ON D.drink_id = DS.drink_id
LEFT JOIN Sweetness AS S ON DS.sw_id = S.sw_id
"""
cursor.execute(query_sw)
data_sw = cursor.fetchall()

ingredients = pd.read_sql_query("SELECT ingr_id, ingr_name FROM Ingredient", conn)

conn.close()


# =========================
# 2) Styles / Helpers
# =========================
COLOR_LIGHT_BROWN = "#EBDEC1"
COLOR_DARK_BROWN = "#5C4033"
COLOR_WHITE = "#FFFFFF"
COLOR_BLACK = "#000000"
FONT_FAMILY = "Arial, sans-serif"
BORDER = False


def pic(name):
    return html.Img(
        src=f"/assets/{name}.png",
        id="drink_pic",
        style={"width": "70%", "marginBottom": "2vw"},
    )


def to_pic(drink):
    # drink may be "xxx (Less Sugar)" or "xxx"
    file = drink.split(" (")[0].replace("'", "").replace(" ", "_").lower()
    return f"/assets/{file}.png"


def find_tp(drink_full_name):
    # drink_full_name looks like: "Pearl Black Tea (Less Sugar)"
    for item in tp_list:
        if item and item != "NULL" and item in drink_full_name:
            return item
    return "NULL"


def block_style(mode):
    if mode == "full":
        return {
            "backgroundColor": COLOR_WHITE,
            "borderRadius": "20px",
            "margin": "2vw 5vw 5vw 5vw",
            "height": "80vh",
            "width": "80vw",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.1)",
        }
    else:
        return {
            "backgroundColor": COLOR_WHITE,
            "borderRadius": "20px",
            "margin": "2vw 2vw 5vw 2vw",
            "height": "80vh",
            "width": "40vw",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.1)",
        }


def word_style(size, alignment):
    return {
        "color": COLOR_DARK_BROWN,
        "fontFamily": FONT_FAMILY,
        "fontSize": f"{size}vw",
        "fontWeight": "normal",
        "textAlign": alignment,
    }


def title_style(size, weight, marginTop, marginBottom):
    return {
        "color": COLOR_DARK_BROWN,
        "fontFamily": FONT_FAMILY,
        "fontSize": f"{size}vw",
        "fontWeight": weight,
        "textAlign": "center",
        "marginTop": f"{marginTop}vw",
        "marginBottom": f"{marginBottom}vw",
    }


def dd_style():
    return {
        "color": COLOR_BLACK,
        "fontFamily": FONT_FAMILY,
        "fontSize": "1vw",
        "fontWeight": "normal",
        "textAlign": "left",
        "border": "1px solid #8B6F47",
        "borderRadius": "5px",
    }


def put_vertical(alignment):
    return {"display": "flex", "flexDirection": "column", "alignItems": alignment}


def put_horizontal(alignment):
    return {"display": "flex", "flexDirection": "row", "alignItems": alignment}


def plot_grid(title):
    return dict(
        title=title,
        showgrid=True,
        gridcolor=COLOR_LIGHT_BROWN,
        gridwidth=1,
        zeroline=True,
        zerolinecolor=COLOR_LIGHT_BROWN,
        zerolinewidth=2,
    )


def title(title1, title2):
    return html.Div(
        [
            html.H1(
                title1,
                style={
                    "margin": "0",
                    "color": COLOR_DARK_BROWN,
                    "fontWeight": "bold",
                    "fontFamily": FONT_FAMILY,
                },
            ),
            html.H2(
                title2,
                style={
                    "margin": "0",
                    "color": COLOR_DARK_BROWN,
                    "fontWeight": "normal",
                    "fontFamily": FONT_FAMILY,
                },
            ),
        ],
        style=put_vertical("center"),
    )


def add_border():
    if BORDER:
        return {"border": "5px dotted red"}
    return {}


# =========================
# 3) Data preprocessing (existing dashboards)
# =========================
drink_list = [item[0] for item in data_drink]

df_ingr = pd.DataFrame(
    data_ingr,
    columns=["drink_name", "base_name", "base_price", "ingr_ml", "ingr_name", "ingr_cal", "ingr_caff"],
)
df_tp = pd.DataFrame(
    data_tp,
    columns=["drink_name", "tp_name", "tp_price", "tp_cal", "tp_caff"],
)
df_tp["tp_name"] = df_tp["tp_name"].fillna("NULL")
df_tp["tp_price"] = df_tp["tp_price"].fillna(0)
df_tp["tp_cal"] = df_tp["tp_cal"].fillna(0)
df_tp["tp_caff"] = df_tp["tp_caff"].fillna(0)

df_sw = pd.DataFrame(data_sw, columns=["drink_name", "sw_name", "sw_cal"])

tp_list = df_tp["tp_name"].unique().tolist()

# --- your original drink_info (with all sweetness combos) for dashboard 3 ---
drink_info_no_sugar = pd.DataFrame({"drink_name": drink_list})

# base & price
price_list = []
base_list = []
for drink in drink_list:
    base_list.append(df_ingr.loc[df_ingr["drink_name"] == drink, "base_name"].iloc[0])
    base_price = df_ingr.loc[df_ingr["drink_name"] == drink, "base_price"].iloc[0]
    tp_price = df_tp.loc[df_tp["drink_name"] == drink, "tp_price"].iloc[0]
    price_list.append(base_price + tp_price)

drink_info_no_sugar["drink_price"] = price_list
drink_info_no_sugar["drink_base"] = base_list

# calories (base ingr + tp)
cal_list = []
for drink in drink_list:
    base_cal = 0
    trim_df = df_ingr.loc[df_ingr["drink_name"] == drink]
    for _, row in trim_df.iterrows():
        base_cal += (row["ingr_cal"] * row["ingr_ml"] / 100)
    tp_cal = df_tp.loc[df_tp["drink_name"] == drink, "tp_cal"].iloc[0]
    cal_list.append(base_cal + tp_cal)
drink_info_no_sugar["drink_cal"] = cal_list

# caffeine (base ingr + tp)
caff_list = []
for drink in drink_list:
    base_caff = 0
    trim_df = df_ingr.loc[df_ingr["drink_name"] == drink]
    for _, row in trim_df.iterrows():
        base_caff += (row["ingr_caff"] * row["ingr_ml"] / 100)
    tp_caff = df_tp.loc[df_tp["drink_name"] == drink, "tp_caff"].iloc[0]
    caff_list.append(base_caff + tp_caff)
drink_info_no_sugar["drink_caff"] = caff_list

# combine sweetness rows
all_drink_list_full = []
all_drink_list = []
all_base_list = []
all_price_list = []
all_cal_list = []
all_caff_list = []
for _, row in df_sw.iterrows():
    all_drink_list_full.append(f"{row['drink_name']} ({row['sw_name']})")
    cor_row = drink_info_no_sugar.loc[
        drink_info_no_sugar["drink_name"] == row["drink_name"],
        ["drink_name", "drink_base", "drink_price", "drink_cal", "drink_caff"],
    ].iloc[0]
    all_drink_list.append(cor_row["drink_name"])
    all_base_list.append(cor_row["drink_base"])
    all_price_list.append(cor_row["drink_price"])
    all_cal_list.append(cor_row["drink_cal"] + row["sw_cal"])
    all_caff_list.append(cor_row["drink_caff"])

drink_info = pd.DataFrame(
    {
        "drink_full_name": all_drink_list_full,
        "drink_name": all_drink_list,
        "drink_base": all_base_list,
        "drink_price": all_price_list,
        "drink_cal": all_cal_list,
        "drink_caff": all_caff_list,
    }
)

# =========================
# 4) Dashboard4: Less Sugar metrics table (fixed sweetness)
# =========================
# base price
base_price_df = df_ingr.groupby("drink_name", as_index=False).agg(base_price=("base_price", "first"))

# base cal/caff from ingredients
base_nutri_df = (
    df_ingr.assign(
        base_cal_part=lambda d: d["ingr_cal"] * d["ingr_ml"] / 100.0,
        base_caff_part=lambda d: d["ingr_caff"] * d["ingr_ml"] / 100.0,
    )
    .groupby("drink_name", as_index=False)
    .agg(base_cal=("base_cal_part", "sum"), base_caff=("base_caff_part", "sum"))
)

# topping aggregation (support multiple toppings)
tp_agg_df = (
    df_tp[df_tp["tp_name"] != "NULL"]
    .groupby("drink_name", as_index=False)
    .agg(tp_price=("tp_price", "sum"), tp_cal=("tp_cal", "sum"), tp_caff=("tp_caff", "sum"), toppings=("tp_name", "count"))
)

# Less Sugar sweetness calories (fallback 0 if missing)
sw_less_df = df_sw[df_sw["sw_name"] == "Less Sugar"][["drink_name", "sw_cal"]].copy()

compare_df = pd.DataFrame({"drink_name": drink_list})
compare_df = compare_df.merge(base_price_df, on="drink_name", how="left")
compare_df = compare_df.merge(base_nutri_df, on="drink_name", how="left")
compare_df = compare_df.merge(tp_agg_df, on="drink_name", how="left")
compare_df = compare_df.merge(sw_less_df, on="drink_name", how="left")

for c in ["base_price", "base_cal", "base_caff", "tp_price", "tp_cal", "tp_caff", "toppings", "sw_cal"]:
    compare_df[c] = compare_df[c].fillna(0)

compare_df["price"] = compare_df["base_price"] + compare_df["tp_price"]
compare_df["calories"] = compare_df["base_cal"] + compare_df["tp_cal"] + compare_df["sw_cal"]
compare_df["caffeine"] = compare_df["base_caff"] + compare_df["tp_caff"]

compare_df = compare_df[["drink_name", "price", "calories", "caffeine", "toppings"]].copy()
drink_options_4 = [{"label": n, "value": n} for n in sorted(compare_df["drink_name"].unique())]

# For dashboard 1 bar chart: use Less Sugar metrics (avoid duplicates)
drink_info_less = compare_df.rename(
    columns={"price": "drink_price", "calories": "drink_cal", "caffeine": "drink_caff"}
)[["drink_name", "drink_price", "drink_cal", "drink_caff"]].copy()


# =========================
# 5) Dashboard 4 Radar (normalize 0–10)
# =========================
RADAR_COLS = {
    "Calories": "calories",
    "Price": "price",
    "Caffeine": "caffeine",
    "Toppings": "toppings",
}


def _transform(metric, x):
    x = float(x)
    if metric in ("Calories", "Price"):
        return float(np.log1p(max(x, 0)))
    return x


_dist = {}
_bounds = {}
for metric, col in RADAR_COLS.items():
    arr = compare_df[col].astype(float).apply(lambda v: _transform(metric, v)).to_numpy()
    arr = np.sort(arr)
    _dist[metric] = arr
    lo = float(np.quantile(arr, 0.05))
    hi = float(np.quantile(arr, 0.95))
    if hi <= lo:
        lo = float(arr.min())
        hi = float(arr.max())
    _bounds[metric] = (lo, hi)


def _percentile_0_10(metric, val):
    arr = _dist[metric]
    if arr.size <= 1:
        return 0.0
    x = _transform(metric, val)
    lo, hi = _bounds[metric]
    x = max(lo, min(x, hi))
    p = np.searchsorted(arr, x, side="left") / (arr.size - 1)
    return 10.0 * float(p)


def _norm_0_10(metric, val):
    if metric in ("Calories", "Price"):
        return _percentile_0_10(metric, val)
    lo, hi = _bounds[metric]
    x = _transform(metric, val)
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
        r = compare_df.loc[compare_df["drink_name"] == drink]
        return None if r.empty else r.iloc[0]

    def raw_values(drink):
        row = get_row(drink)
        vals = [None] * len(cats) if row is None else [float(row[RADAR_COLS[c]]) for c in cats]
        return vals + [vals[0]]

    def scaled_values(drink):
        row = get_row(drink)
        vals = [0.0] * len(cats) if row is None else [_norm_0_10(c, row[RADAR_COLS[c]]) for c in cats]
        return vals + [vals[0]]

    def fmt(metric, v):
        if v is None:
            return "—"
        if metric == "Price":
            return f"{v:.0f}"
        return f"{v:.0f}"

    left_raw = raw_values(left_drink)
    right_raw = raw_values(right_drink)
    left_name = left_drink or ""
    right_name = right_drink or ""

    hover_text = []
    for i, m in enumerate(cats_closed):
        hover_text.append(
            f"<b>Metric:</b> {m}<br>"
            f"<b>{left_name}:</b> {fmt(m, left_raw[i])}<br>"
            f"<b>{right_name}:</b> {fmt(m, right_raw[i])}"
        )

    fig = go.Figure()

    # draw right then left (left on top -> hover more stable)
    fig.add_trace(
        go.Scatterpolar(
            r=scaled_values(right_drink),
            theta=cats_closed,
            fill="toself",
            name=right_name,
            mode="lines+markers",
            opacity=0.85,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=scaled_values(left_drink),
            theta=cats_closed,
            fill="toself",
            name=left_name,
            mode="lines+markers",
            opacity=0.85,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    fig.update_layout(
        paper_bgcolor=COLOR_WHITE,
        plot_bgcolor=COLOR_WHITE,
        font=dict(family=FONT_FAMILY, size=14, color=COLOR_DARK_BROWN),
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="closest",
        polar=dict(
            domain=dict(x=[0.08, 0.92], y=[0.08, 0.92]),  # avoid overflow
            radialaxis=dict(visible=True, range=[0, 10], showticklabels=False, gridcolor=COLOR_LIGHT_BROWN),
            angularaxis=dict(gridcolor=COLOR_LIGHT_BROWN, tickfont=dict(size=12), rotation=90, direction="clockwise"),
        ),
    )
    return fig


# =========================
# 6) Dashboard elements
# =========================
# Dashboard 1
bar_dd = dcc.Dropdown(
    id="metric-dropdown",
    options=[
        {"label": "Kcal (熱量)", "value": "drink_cal"},
        {"label": "Price (價格)", "value": "drink_price"},
        {"label": "Caffeine (咖啡因)", "value": "drink_caff"},
    ],
    value="drink_cal",
    clearable=False,
    style={"width": "200px"},
)
bar_radiobtn = dcc.RadioItems(
    id="sort-order",
    options=[
        {"label": "前 10 名 (最高)", "value": "descending"},
        {"label": "最後 10 名 (最低)", "value": "ascending"},
    ],
    value="descending",
)
bar_select = html.Div(
    [
        html.Label("選擇比較項目:", style={"color": COLOR_DARK_BROWN, "fontWeight": "bold"}),
        bar_dd,
        html.Label("排序方式:", style={"color": COLOR_DARK_BROWN, "fontWeight": "bold"}),
        bar_radiobtn,
    ],
    style={"display": "flex", "alignItems": "center", "justifyContent": "center"},
)

# Dashboard 2
table_dd = html.Div(
    [
        html.Div(
            "Select Ingredient:",
            style={"fontWeight": "500", "display": "block", "marginBottom": "1vw", **word_style("1", "left")},
        ),
        dcc.Dropdown(
            id="ingredient",
            options=[{"label": row["ingr_name"], "value": row["ingr_id"]} for _, row in ingredients.iterrows()],
            placeholder="All...",
            style={**dd_style(), "width": "250px"},
        ),
    ],
    style={"marginLeft": "35px", "marginTop": "35px"},
)
table = html.Div([html.Div(id="output")], style={"padding": "35px", "minHeight": "400px"})

# Dashboard 3 (your existing scatter dashboard)
sw_dd = dcc.Dropdown(
    id="sw_dd",
    options=[
        {"label": "Full Sugar (100%)", "value": "Full Sugar"},
        {"label": "Half Sugar (50%)", "value": "Half Sugar"},
        {"label": "Less Sugar (20%)", "value": "Less Sugar"},
        {"label": "No Sugar (0%)", "value": "No Sugar"},
    ],
    value="Select a sweetness",
    style={**dd_style()},
)
base_dd = dcc.Dropdown(id="base_dd", options=base_list, value="Select a base", style={**dd_style()})
tp_dd = dcc.Dropdown(id="tp_dd", options=tp_list, value="Select a topping", style=dd_style())

scatter_fig = px.scatter(
    data_frame=drink_info,
    x="drink_cal",
    y="drink_price",
    color="drink_name",
    hover_data=["drink_full_name"],
    custom_data=["drink_full_name"],
    range_x=[0, 850],
    range_y=[20, 85],
)
scatter_fig.update_layout(
    font=dict(family=FONT_FAMILY, size=16, color=COLOR_DARK_BROWN),
    plot_bgcolor=COLOR_WHITE,
    xaxis=plot_grid("Calories (Kcal)"),
    yaxis=plot_grid("Price (NTD)"),
    showlegend=False,
)

scatter_plot = dcc.Graph(id="scatter_plot", figure=scatter_fig, style={"width": "100%", "height": "100%"})

scatter_dd = html.Div(
    children=[
        html.Div([html.Div("Sweetness", style={**word_style("1", "left"), "marginBottom": "0.5vw"}), sw_dd]),
        html.Div([html.Div("Base", style={**word_style("1", "left"), "marginBottom": "0.5vw"}), base_dd]),
        html.Div([html.Div("Topping", style={**word_style("1", "left"), "marginBottom": "0.5vw"}), tp_dd]),
    ],
    style={"width": "12vw", **put_vertical("left"), **add_border(), "gap": "1vw"},
)

scatter_cup = html.Div(
    children=[
        pic("cup"),
        html.Div("", id="name_text", style={"height": "2vw", **word_style("1", "center"), "width": "85%"}),
    ],
    style={"width": "12vw", **put_vertical("center"), **add_border(), "marginLeft": "1vw"},
)

scatter_big_text = html.Div(
    children=[
        html.Div("Price", style={**word_style("1.5", "center"), "marginBottom": "1vw"}),
        html.Div("--", id="price_text", style={**word_style("3", "center"), "marginBottom": "3vw"}),
        html.Div("Calories", style={**word_style("1.5", "center"), "marginBottom": "1vw"}),
        html.Div("--", id="cal_text", style=word_style("3", "center")),
    ],
    style={"width": "12vw", **put_vertical("center"), **add_border()},
)

# Dashboard 4 UI
dash4_block = html.Div(
    style={**block_style("full"), "overflow": "hidden"},
    children=[
        html.Div(
            style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between",
                "gap": "2vw",
                "height": "70vh",
                "padding": "2vw",
            },
            children=[
                html.Div(
                    style={"width": "24%", "display": "flex", "flexDirection": "column", "alignItems": "center", "gap": "1vw"},
                    children=[
                        dcc.Dropdown(id="left_dd", options=drink_options_4, placeholder="Select Drink", value=None, style={"width": "90%"}),
                        html.Img(id="left_img", src="/assets/cup.png", style={"width": "70%"}),
                        html.Div(id="left_name", style={**word_style("1", "center"), "minHeight": "2vw"}),
                    ],
                ),
                html.Div(
                    style={"width": "45%", "height": "100%"},
                    children=[
                        dcc.Graph(
                            id="radar",
                            figure=radar_fig(None, None),
                            style={"width": "100%", "height": "62vh"},
                            config={"displayModeBar": False},
                        )
                    ],
                ),
                html.Div(
                    style={"width": "24%", "display": "flex", "flexDirection": "column", "alignItems": "center", "gap": "1vw"},
                    children=[
                        dcc.Dropdown(id="right_dd", options=drink_options_4, placeholder="Select Drink", value=None, style={"width": "90%"}),
                        html.Img(id="right_img", src="/assets/cup.png", style={"width": "70%"}),
                        html.Div(id="right_name", style={**word_style("1", "center"), "minHeight": "2vw"}),
                    ],
                ),
            ],
        ),
        html.Div(
            "* Default sweetness used: Less Sugar (Calories includes sw_cal at Less Sugar).",
            style={"marginTop": "0.5vw", "textAlign": "center", "color": COLOR_DARK_BROWN, "fontFamily": FONT_FAMILY},
        ),
    ],
)


# =========================
# 7) App layout
# =========================
app = Dash()

app.layout = html.Div(
    children=html.Div(
        [
            html.Div(
                [
                    html.Div("The Perfect Cup Finder", style=title_style(2.5, 900, 2, 0.5)),
                    html.Div("Your Drink Decision Guide", style=title_style(2, 700, 0, 4)),
                ],
                style=put_vertical("center"),
            ),
            # Dashboard 1 + 2
            html.Div(
                [
                    html.Div(
                        [
                            title("Start Your Drink Journey:", "What's Trending on the Menu?"),
                            html.Div([bar_select, dcc.Graph(id="bar-chart")], style={**block_style("half"), **put_vertical("center")}),
                        ],
                        style=put_vertical("center"),
                    ),
                    html.Div(
                        [
                            title("Choose Your Flavor Base:", "Discover Drinks Made Just for You"),
                            html.Div([table_dd, table], style={**block_style("half"), **put_vertical("left")}),
                        ],
                        style=put_vertical("center"),
                    ),
                ],
                style=put_horizontal("center"),
            ),
            # Dashboard 3
            html.Div(
                [
                    title("Build Your Perfect Cup:", "Customize Sweetness, Base, and Toppings"),
                    html.Div(
                        [
                            dcc.Store(id="base_options_store", data=pd.Series(base_list).unique().tolist()),
                            dcc.Store(id="tp_options_store", data=tp_list),
                            scatter_dd,
                            scatter_cup,
                            scatter_big_text,
                            html.Div(scatter_plot, style={"width": "40vw", "height": "100%", **add_border()}),
                        ],
                        style={**put_horizontal("center"), **block_style("full"), "padding": "0vw 0vw 0vw 5vw", "gap": "0.5vw"},
                    ),
                ],
                style=put_vertical("center"),
            ),
            # Dashboard 4
            html.Div(
                [
                    title("Final Showdown:", "Compare Your Top Picks Before You Sip!"),
                    dash4_block,
                ],
                style=put_vertical("center"),
            ),
        ],
        style=put_vertical("center"),
    ),
    style={"backgroundColor": COLOR_LIGHT_BROWN, "minHeight": "100vh", "paddingBottom": "5vw"},
)


# =========================
# 8) Callbacks
# =========================
# Dashboard 1 callback (use Less Sugar metrics to avoid duplicates)
@callback(
    Output("bar-chart", "figure"),
    Input("metric-dropdown", "value"),
    Input("sort-order", "value"),
)
def update_graph(selected_metric, sort_order):
    is_ascending = True if sort_order == "ascending" else False
    sorted_df = drink_info_less.sort_values(by=selected_metric, ascending=is_ascending)
    top_10_df = sorted_df.head(10)

    labels_map = {
        "drink_cal": "熱量 (Kcal) [Less Sugar]",
        "drink_price": "價格 (NTD)",
        "drink_caff": "咖啡因 (mg)",
        "drink_name": "飲料名稱",
    }

    fig = px.bar(
        top_10_df,
        x="drink_name",
        y=selected_metric,
        text=selected_metric,
        labels=labels_map,
        title=f"飲料{labels_map[selected_metric]}排名 ({'最低' if is_ascending else '最高'} 10 名)",
    )

    fig.update_traces(textposition="outside", marker_color=COLOR_DARK_BROWN)
    fig.update_layout(
        xaxis={"categoryorder": "total ascending"} if is_ascending else {"categoryorder": "total descending"},
        plot_bgcolor="white",
        font_color=COLOR_DARK_BROWN,
        title_font_color=COLOR_DARK_BROWN,
        title_x=0.5,
    )
    return fig


# Dashboard 2 callback (your original logic)
@callback(Output("output", "children"), Input("ingredient", "value"))
def show_drinks(ingr_id):
    conn = sqlite3.connect(DB_PATH)

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

    df["tp_price"] = df["tp_price"].fillna(0)
    df["sw_cal"] = df["sw_cal"].fillna(0)
    df["tp_cal"] = df["tp_cal"].fillna(0)
    df["tp_caff"] = df["tp_caff"].fillna(0)

    df = df.merge(df_base[["drink_name", "base_cal", "base_caff"]], on="drink_name", how="left")
    df["base_cal"] = df["base_cal"].fillna(0)
    df["base_caff"] = df["base_caff"].fillna(0)

    df["Price"] = (df["base_price"] + df["tp_price"]).astype(int)
    df["Calories"] = (df["base_cal"] + df["sw_cal"] + df["tp_cal"]).astype(int)
    df["Caffeine"] = (df["base_caff"] + df["tp_caff"]).astype(int)

    result = df[["drink_name", "Price", "Calories", "Caffeine"]].copy()
    result.columns = ["Name", "Price", "Kcal", "Caffeine(mg)"]
    result = result.drop_duplicates()

    return html.Div(
        [
            dash_table.DataTable(
                data=result.to_dict("records"),
                columns=[{"name": i, "id": i} for i in result.columns],
                sort_action="native",
                page_action="native",
                page_size=10,
                style_header={
                    "backgroundColor": "#8B6F47",
                    "color": "white",
                    "fontWeight": "bold",
                    "fontSize": "14px",
                    "textAlign": "center",
                    "padding": "12px",
                    "border": "none",
                },
                style_cell={
                    "textAlign": "center",
                    "padding": "12px 10px",
                    "backgroundColor": "white",
                    "border": "none",
                    "borderBottom": "1px solid #E7DABD",
                    "fontSize": "14px",
                    "color": "#333",
                },
                style_cell_conditional=[
                    {"if": {"column_id": "Name"}, "textAlign": "left", "fontWeight": "500", "paddingLeft": "15px"}
                ],
                style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#F4F0E9"}],
            ),
            html.Div(
                "* Calories are calculated based on Less Sugar level.",
                style={"marginTop": "20px", "fontStyle": "italic", "color": "#999", "fontSize": "13px", "textAlign": "center"},
            ),
        ]
    )


# Dashboard 3 callback (your original logic)
@callback(
    Output("scatter_plot", "figure"),
    Output("drink_pic", "src"),
    Output("price_text", "children"),
    Output("cal_text", "children"),
    Output("name_text", "children"),
    Output("base_dd", "options"),
    Output("tp_dd", "options"),
    Output("base_options_store", "data"),
    Output("tp_options_store", "data"),
    Input("sw_dd", "value"),
    Input("base_dd", "value"),
    Input("tp_dd", "value"),
    Input("scatter_plot", "clickData"),
    State("base_options_store", "data"),
    State("tp_options_store", "data"),
)
def update_scatter(sw, base, tp, clickData, base_options_data, tp_options_data):
    ctx = callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

    img = "/assets/cup.png"
    price = "--"
    cal = "--"
    name = ""
    base_options = list(base_options_data) if base_options_data else pd.Series(base_list).unique().tolist()
    tp_options = list(tp_options_data) if tp_options_data else tp_list

    new_drink_info = drink_info.copy()

    if sw and sw != "Select a sweetness":
        new_drink_info = new_drink_info.loc[new_drink_info["drink_full_name"].str.contains(sw)]
    if base and base != "Select a base":
        new_drink_info = new_drink_info.loc[new_drink_info["drink_base"] == base]
    if tp and tp != "Select a topping":
        drinks_with_tp = df_tp.loc[df_tp["tp_name"] == tp, "drink_name"].unique()
        new_drink_info = new_drink_info[new_drink_info["drink_name"].isin(drinks_with_tp)]

    new_fig = px.scatter(
        data_frame=new_drink_info,
        x="drink_cal",
        y="drink_price",
        color="drink_name",
        hover_data=["drink_full_name"],
        custom_data=["drink_full_name"],
        range_x=[-10, 810],
        range_y=[28, 82],
    )
    new_fig.update_layout(
        font=dict(family=FONT_FAMILY, size=16, color=COLOR_DARK_BROWN),
        plot_bgcolor=COLOR_WHITE,
        xaxis=plot_grid("Calories (Kcal)"),
        yaxis=plot_grid("Price (NTD)"),
        showlegend=False,
    )

    if trigger_id == "sw_dd" and sw and sw != "Select a sweetness":
        if len(new_drink_info["drink_name"].unique().tolist()) == 1:
            img = to_pic(new_drink_info["drink_name"].unique().tolist()[0])
        elif base and base != "Select a base":
            img = to_pic(base)
        elif tp and tp not in ("NULL", "Select a topping"):
            img = to_pic(tp)

    elif trigger_id == "base_dd" and base and base != "Select a base":
        if len(new_drink_info["drink_name"].unique().tolist()) == 1:
            img = to_pic(new_drink_info["drink_name"].unique().tolist()[0])
        else:
            img = to_pic(base)

        tp_options = []
        for _, row in drink_info.iterrows():
            current_tp = find_tp(row["drink_full_name"])
            current_base = row["drink_base"]
            if current_base == base and current_tp not in tp_options:
                tp_options.append(current_tp)

    elif trigger_id == "tp_dd" and tp and tp != "Select a topping":
        if len(new_drink_info["drink_name"].unique().tolist()) == 1:
            img = to_pic(new_drink_info["drink_name"].unique().tolist()[0])
        elif tp != "NULL":
            img = to_pic(tp)

        base_options = []
        for _, row in drink_info.iterrows():
            current_tp = find_tp(row["drink_full_name"])
            current_base = row["drink_base"]
            if current_tp == tp and current_base not in base_options:
                base_options.append(current_base)

    elif trigger_id == "scatter_plot" and clickData is not None:
        name = clickData["points"][0]["customdata"][0]
        price = drink_info.loc[drink_info["drink_full_name"] == name, "drink_price"].iloc[0]
        cal = drink_info.loc[drink_info["drink_full_name"] == name, "drink_cal"].iloc[0]
        img = to_pic(name)

    # reset when both cleared
    if (not base or base == "Select a base") and (not tp or tp == "Select a topping"):
        base_options = pd.Series(base_list).unique().tolist()
        tp_options = tp_list

    return new_fig, img, price, cal, name, base_options, tp_options, base_options, tp_options


# Dashboard 4 callback
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

    return fig, left_src, (left_drink or ""), right_src, (right_drink or "")


if __name__ == "__main__":
    app.run(debug=True)

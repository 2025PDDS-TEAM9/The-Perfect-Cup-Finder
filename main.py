import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from dash import Dash, dcc, html, Input, Output, callback, callback_context

# connect db and get data
conn = sqlite3.connect('Tea_Shop_Database.db')
cursor = conn.cursor()

# Drink list
cursor.execute("SELECT drink_name FROM Drink")
data_drink = cursor.fetchall()

# Ingredients (base + ingredient composition)
query_ingr = """
SELECT D.drink_name, B.base_name, B.base_price, BI.ingr_ml, I.ingr_name, I.ingr_cal, I.ingr_caff
FROM Drink AS D
JOIN Base AS B ON D.base_id = B.base_id
JOIN BaseIngredient AS BI ON B.base_id = BI.base_id
JOIN Ingredient AS I ON BI.ingr_id = I.ingr_id
"""
cursor.execute(query_ingr)
data_ingr = cursor.fetchall()

# Toppings (can be multiple rows per drink)
query_tp = """
SELECT D.drink_name, T.tp_name, T.tp_price, T.tp_cal, T.tp_caff
FROM Drink AS D
LEFT JOIN DrinkTopping AS DT ON D.drink_id = DT.drink_id
LEFT JOIN Topping AS T ON DT.tp_id = T.tp_id
"""
cursor.execute(query_tp)
data_tp = cursor.fetchall()

# Sweetness (multiple rows per drink)
query_sw = """
SELECT D.drink_name, S.sw_name, S.sw_cal
FROM Drink AS D
LEFT JOIN DrinkSweetness AS DS ON D.drink_id = DS.drink_id
LEFT JOIN Sweetness AS S ON DS.sw_id = S.sw_id
"""
cursor.execute(query_sw)
data_sw = cursor.fetchall()

conn.close()


# 2) Styles
COLOR_LIGHT_BROWN = "#EBDEC1"
COLOR_DARK_BROWN = "#5C4033"
COLOR_WHITE = "#FFFFFF"

def word_style(size, alignment):
    return {
        "color": COLOR_DARK_BROWN,
        "fontFamily": "Arial, sans-serif",
        "fontSize": f"{size}vw",
        "fontWeight": "normal",
        "textAlign": alignment,
    }

def to_pic(drink_name: str) -> str:
    file = drink_name.replace("'", "").replace(" ", "_").lower()
    return f"/assets/{file}.png"

# 3) Data preprocessing
drink_list = [x[0] for x in data_drink]

df_ingr = pd.DataFrame(
    data_ingr,
    columns=["drink_name", "base_name", "base_price", "ingr_ml", "ingr_name", "ingr_cal", "ingr_caff"],
)

df_tp = pd.DataFrame(
    data_tp,
    columns=["drink_name", "tp_name", "tp_price", "tp_cal", "tp_caff"],
)
df_tp["tp_name"] = df_tp["tp_name"].fillna("NULL")
df_tp[["tp_price", "tp_cal", "tp_caff"]] = df_tp[["tp_price", "tp_cal", "tp_caff"]].fillna(0)

df_sw = pd.DataFrame(
    data_sw,
    columns=["drink_name", "sw_name", "sw_cal"],
)
df_sw = df_sw.dropna(subset=["sw_name"])  # 去掉沒有 sweetness 的 row


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
drink_info = drink_base.merge(df_sw_default, on="drink_name", how="left")

# Price = base_price + Σ(tp_price)
drink_info["price"] = drink_info["base_price"] + drink_info["tp_price"]

# Calories = Σ(ingr_cal*ml/100) + Σ(tp_cal) + sw_cal
drink_info["calories"] = drink_info["base_cal"] + drink_info["tp_cal"] + drink_info["sw_cal"]

# Caffeine = Σ(ingr_caff*ml/100) + Σ(tp_caff)
drink_info["caffeine"] = drink_info["base_caff"] + drink_info["tp_caff"]

# Keep only needed columns
plot_df = drink_info[["drink_name", "price", "calories", "caffeine", "toppings", "sw_name"]].copy()



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
        font=dict(family="Arial, sans-serif", size=14, color=COLOR_DARK_BROWN),
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
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
                tickfont=dict(size=12),
                rotation=90,
                direction="clockwise",
            ),
        ),
    )
    return fig

# 5) Dash layout (Compare only)
app = Dash(__name__)

drink_options = [{"label": n, "value": n} for n in sorted(plot_df["drink_name"].unique())]

app.layout = html.Div(
    style={"backgroundColor": COLOR_LIGHT_BROWN, "minHeight": "100vh", "padding": "2.5vw"},
    children=[
        html.H2(
            "Final Showdown: Compare Your Top Picks Before You Sip!",
            style={"textAlign": "center", "color": COLOR_DARK_BROWN, "marginBottom": "2vw", "fontFamily": "Arial"},
        ),

        html.Div(
            style={
                "backgroundColor": COLOR_WHITE,
                "borderRadius": "24px",
                "padding": "2.5vw 3vw",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between",
                "gap": "2vw",
                "height": "75vh",
            },
            children=[
                # Left
                html.Div(
                    style={"width": "24%", "display": "flex", "flexDirection": "column", "alignItems": "center", "gap": "1vw"},
                    children=[
                        dcc.Dropdown(
                            id="left_dd",
                            options=drink_options,
                            placeholder="Select Item",
                            value=None,
                            style={"width": "90%"},
                        ),
                        html.Img(id="left_img", src="/assets/cup.png", style={"width": "70%"}),
                        html.Div(id="left_name", style={**word_style("1.0", "center"), "minHeight": "2vw"}),
                    ],
                ),

                # Radar
                html.Div(
                    style={"width": "45%", "height": "100%"},
                    children=[
                        dcc.Graph(
                            id="radar",
                            figure=radar_fig(None, None),
                            style={"width": "100%", "height": "100%"},
                        )
                    ],
                ),

                # Right
                html.Div(
                    style={"width": "24%", "display": "flex", "flexDirection": "column", "alignItems": "center", "gap": "1vw"},
                    children=[
                        dcc.Dropdown(
                            id="right_dd",
                            options=drink_options,
                            placeholder="Select Item",
                            value=None,
                            style={"width": "90%"},
                        ),
                        html.Img(id="right_img", src="/assets/cup.png", style={"width": "70%"}),
                        html.Div(id="right_name", style={**word_style("1.0", "center"), "minHeight": "2vw"}),
                    ],
                ),
            ],
        ),

        # Optional: 顯示預設甜度是什麼
        html.Div(
            f"Default sweetness for all drinks: Less Sugar (fallback to first available if missing).",
            style={"marginTop": "1vw", **word_style("0.9", "center")},
        ),
    ],
)

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


if __name__ == "__main__":
    app.run(debug=True)
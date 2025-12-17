import dash
from dash import dcc, html, dash_table, Input, Output
import pandas as pd
import sqlite3

# 連接資料庫
DB = 'Tea_Shop_Database.db'
conn = sqlite3.connect(DB)  
ingredients = pd.read_sql_query("SELECT ingr_id, ingr_name FROM Ingredient", conn)
conn.close() 

# Dash 
app = dash.Dash(__name__)  

app.layout = html.Div([
    # 外層（米色背景）
    html.Div([
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
        })

    ], style={
        'backgroundColor': "#EBDEC1",
        'minHeight': '100vh',
        'padding': '40px 20px',
        'fontFamily': 'Arial, sans-serif'
    })
])

# Callback
@app.callback(  
    Output('output', 'children'),  
    Input('ingredient', 'value')  
)
def show_drinks(ingr_id):
    
    conn = sqlite3.connect(DB)
    
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

if __name__ == '__main__':
    app.run(debug=True)
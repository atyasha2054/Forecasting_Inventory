import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
from datetime import datetime, timedelta
import dash_daq as daq
from dash import dash_table
import random
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, IsolationForest
import numpy as np

# Sample Inventory Data (Hardcoded)
inventory = pd.DataFrame({
    "item": ["rice", "wheat", "pulses", "sugar"],
    "stock": [100, 150, 60, 40],
    "consumption_rate": [0.5, 0.7, 0.6, 0.2],
    "threshold": [20, 30, 10, 10],
    "last_replenished": [None, None, None, None]
})

# Helper function to format the date
def format_date(date):
    if date:
        return date.strftime('%Y-%m-%d %H:%M:%S')
    return "Not available"

# Function to forecast ration required for N days
def forecast_ration(n_days, n_people):
    inventory_copy = inventory.copy()
    inventory_copy["required_ration"] = inventory_copy["consumption_rate"] * n_days * n_people
    inventory_copy["sufficient_stock"] = inventory_copy["stock"] >= inventory_copy["required_ration"]
    return inventory_copy[["item", "stock", "required_ration", "sufficient_stock"]]

def replenish_if_needed():
    global inventory
    for i in range(len(inventory)):
        item_name = inventory.loc[i, "item"]
        current_stock = inventory.loc[i, "stock"]
        threshold = inventory.loc[i, "threshold"]
        if current_stock <= threshold:
            historical_consumption_data = [5, 6, 7, 8, 7, 6]
            predicted_consumption = predict_consumption(item_name, historical_consumption_data)
            if predicted_consumption:
                target_days = 30
                required_stock = predicted_consumption * target_days
                replenishment_amount = max(0, required_stock - current_stock)
                if replenishment_amount > 0:
                    inventory.loc[i, "stock"] += replenishment_amount
                    inventory.loc[i, "last_replenished"] = datetime.now()
                    replenishment_amount_round = round(replenishment_amount)
                    print(f"\nReplenishing {item_name} by {replenishment_amount:.2f} units.")
                else:
                    print(f"No replenishment needed for '{item_name}'.")
            else:
                print(f"Could not predict replenishment for '{item_name}'.")
        else:
            print(f"'{item_name}' has sufficient stock.")

def simulate_consumption(n_people, n_days):
    print(f"\nSimulating consumption for {n_people} people over {n_days} days...")
    simulation_log = []
    for day in range(1, n_days + 1):
        day_log = {"day": day}
        for idx, row in inventory.iterrows():
            daily_consumption = row['consumption_rate'] * n_people
            adjusted_consumption = min(daily_consumption, row['stock'])
            inventory.at[idx, 'stock'] -= adjusted_consumption
            day_log[row['item']] = inventory.at[idx, 'stock']
        simulation_log.append(day_log)
        low_stock_alert(5)
        replenish_if_needed()
        if any(inventory["stock"] <= 0):
            break
    return pd.DataFrame(simulation_log)

def low_stock_alert(days_in_advance=5):
    global inventory
    inventory["predicted_days_left"] = inventory.apply(
        lambda row: row["stock"] / (row["consumption_rate"] * 1.5), axis=1)
    low_stock_items = inventory[inventory["predicted_days_left"] <= days_in_advance]
    if not low_stock_items.empty:
        print(f"\nLow Stock Alert (Items running low within {days_in_advance} days):")
        print(low_stock_items[["item", "stock", "predicted_days_left"]])
    else:
        print(f"\nNo items are running low within the next {days_in_advance} days.")

def predict_consumption(item_name, historical_data):
    if item_name in inventory['item'].values:
        days = np.array(range(len(historical_data))).reshape(-1, 1)
        consumption = np.array(historical_data)
        model = RandomForestRegressor(n_estimators=100)
        model.fit(days, consumption)
        next_day = np.array([[len(historical_data)]])
        predicted_consumption = model.predict(next_day)[0]
        print(f"Predicted consumption rate for '{item_name}' on the next day: {predicted_consumption:.2f} units")
        return predicted_consumption
    else:
        print(f"Item '{item_name}' not found in inventory.")
        return None

def forecast_depletion():
    today = datetime.now()
    depletion_forecast = []
    for idx, row in inventory.iterrows():
        predicted_consumption = predict_consumption(row['item'], [5, 6, 7, 8, 7, 6])
        depletion_time = row['stock'] / predicted_consumption if predicted_consumption else row['stock'] / row['consumption_rate']
        depletion_date = today + timedelta(days=depletion_time)
        depletion_forecast.append({"item": row['item'], "depletion_date": depletion_date.strftime('%Y-%m-%d')})
    return pd.DataFrame(depletion_forecast)

# Dash App
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Inventory Management Dashboard"

app.layout = html.Div([
    html.Div([
        html.H1("Inventory Management Dashboard", style={
            'textAlign': 'center', 'fontSize': '3em', 'color': '#ffffff', 'marginBottom': '20px',
            'fontFamily': 'Roboto, sans-serif', 'fontWeight': '700'}),
        dcc.Tabs([
            dcc.Tab(label="Forecast Ration", children=[
                html.Div([
                    html.Div([
                        html.Div("Instructions: Forecast ration for N days and N people. Enter values below.", style={
                            'fontSize': '1em', 'color': '#ffffff', 'marginBottom': '20px'}),
                        html.Label("Number of Days:", style={'color': '#ffffff'}),
                        dcc.Input(id="forecast_days", type="number", min=1, value=7,
                                  style={'width': '200px', 'margin': '10px'}),
                        html.Label("Number of People:", style={'color': '#ffffff'}),
                        dcc.Input(id="forecast_people", type="number", min=1, value=10,
                                  style={'width': '200px', 'margin': '10px'}),
                        html.Button("Forecast", id="forecast_button", n_clicks=0, style={
                            'backgroundColor': '#4CAF50', 'color': 'white', 'padding': '10px 20px',
                            'borderRadius': '4px', 'cursor': 'pointer'}),
                    ], style={'marginBottom': '20px'}),
                    html.Div(id="forecast_output", style={'marginTop': '20px'})
                ], style={'padding': '20px', 'backgroundColor': '#2c2c2c'})
            ]),
            dcc.Tab(label="Simulate Consumption", children=[
                html.Div([
                    html.Div("Simulate consumption for N people over multiple days.", style={
                        'color': '#ffffff', 'marginBottom': '20px'}),
                    html.Label("Number of People:", style={'color': '#ffffff'}),
                    dcc.Input(id="simulate_people", type="number", min=1, value=10,
                              style={'width': '200px', 'margin': '10px'}),
                    html.Label("Number of Days:", style={'color': '#ffffff'}),
                    dcc.Input(id="simulate_days", type="number", min=1, value=7,
                              style={'width': '200px', 'margin': '10px'}),
                    html.Button("Simulate", id="simulate_button", n_clicks=0, style={
                        'backgroundColor': '#4CAF50', 'color': 'white', 'padding': '10px 20px',
                        'borderRadius': '4px', 'cursor': 'pointer'}),
                    html.Div(id="simulate_output", style={'marginTop': '20px'})
                ], style={'padding': '20px', 'backgroundColor': '#2c2c2c'})
            ]),
            dcc.Tab(label="Forecast Depletion", children=[
                html.Div([
                    html.Div("Forecast depletion dates based on current stock.", style={
                        'color': '#ffffff', 'marginBottom': '20px'}),
                    html.Button("Forecast Depletion", id="depletion_button", n_clicks=0, style={
                        'backgroundColor': '#4CAF50', 'color': 'white', 'padding': '10px 20px',
                        'borderRadius': '4px', 'cursor': 'pointer'}),
                    html.Div(id="depletion_output", style={'marginTop': '20px'})
                ], style={'padding': '20px', 'backgroundColor': '#2c2c2c'})
            ])
        ])
    ], style={
        'maxWidth': '1200px', 'margin': '0 auto', 'padding': '20px',
        'backgroundColor': '#1e1e1e', 'boxShadow': '0px 4px 6px rgba(0, 0, 0, 0.1)', 'borderRadius': '10px'
    })
])

# Callbacks
@app.callback(
    Output("forecast_output", "children"),
    Input("forecast_button", "n_clicks"),
    State("forecast_days", "value"),
    State("forecast_people", "value")
)
def update_forecast(n_clicks, n_days, n_people):
    if n_clicks > 0:
        forecast = forecast_ration(n_days, n_people)
        return dash_table.DataTable(
            data=forecast.to_dict('records'),
            columns=[{'name': i.capitalize(), 'id': i} for i in forecast.columns],
            style_table={'width': '100%', 'marginTop': '10px'},
            style_cell={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#2c2c2c', 'color': 'white'},
            style_header={'backgroundColor': '#4CAF50', 'color': 'white', 'fontWeight': 'bold'},
        )
    return html.Div("Click the button to forecast ration.", style={'color': 'white', 'marginTop': '10px'})

@app.callback(
    Output("simulate_output", "children"),
    Input("simulate_button", "n_clicks"),
    State("simulate_people", "value"),
    State("simulate_days", "value")
)
def update_simulation(n_clicks, n_people, n_days):
    if n_clicks > 0:
        simulation = simulate_consumption(n_people, n_days)
        return dash_table.DataTable(
            data=simulation.to_dict('records'),
            columns=[{'name': i.capitalize(), 'id': i} for i in simulation.columns],
            style_table={'width': '100%', 'marginTop': '10px'},
            style_cell={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#2c2c2c', 'color': 'white'},
            style_header={'backgroundColor': '#4CAF50', 'color': 'white', 'fontWeight': 'bold'},
        )

@app.callback(
    Output("depletion_output", "children"),
    Input("depletion_button", "n_clicks")
)
def update_depletion(n_clicks):
    if n_clicks > 0:
        depletion = forecast_depletion()
        return dash_table.DataTable(
            data=depletion.to_dict('records'),
            columns=[{'name': i.capitalize(), 'id': i} for i in depletion.columns],
            style_table={'width': '100%', 'marginTop': '10px'},
            style_cell={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#2c2c2c', 'color': 'white'},
            style_header={'backgroundColor': '#4CAF50', 'color': 'white', 'fontWeight': 'bold'},
        )

# Expose the app object for Gunicorn
server = app.server

# Run the app
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=os.environ.get("PORT", 8050))

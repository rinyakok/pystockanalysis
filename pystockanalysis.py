"""This script implements an interactive stock analysis dashboard using Dash, Plotly, and yfinance. 
It provides visualization and technical analysis tools for a selected stock, including candlestick charts, RSI, MACD, and dynamic trendline detection.
Detailed Description:
---------------------
The application fetches historical stock data for a specified ticker (default: "OTP.BD") using yfinance. 
It computes technical indicators such as the Relative Strength Index (RSI) and the Moving Average Convergence Divergence (MACD) and visualizes them using Plotly charts.
Key Features:
-------------
1. **Candlestick Chart**: Visualizes the stock's open, high, low, and close prices over the last year.
2. **RSI and MACD Charts**: Plots the RSI (14-period) and MACD (12, 26, 9) indicators for momentum and trend analysis.
3. **Date Range Selection**: Users can select a date range using a RangeSlider, which updates all charts to highlight the selected period.
4. **Trendline Calculation**:
    - **Fast Trendline**: Computes a simple linear regression (least squares fit) over the selected range.
    - **Optimized Support/Resistance Trendlines**: 
        - Identifies the most extreme support (lowest low) and resistance (highest high) points relative to the trendline.
        - Iteratively adjusts the trendline slope to pass through these points and maximize intersection with lows (support) or highs (resistance).
        - Trendlines are extended ("prolonged") 20 days beyond the selected range for forward-looking analysis.
5. **Interactive Controls**: 
    - Checklist to toggle trendline visibility.
    - Visual feedback on selected date range.
    - All charts update dynamically based on user input.
How It Works:
-------------
- The script defines several helper functions for technical analysis:
    - `calculate_rsi`: Computes the RSI for a price series.
    - `fast_trendline`: Fits a linear trendline to the 'Close' price between two indices.
    - `find_extreme_point`: Locates the most extreme support or resistance point near the trendline.
    - `count_trendline_intersections`: Counts how many lows/highs intersect the trendline within a precision threshold.
    - `optimize_trendline_intersections`: Iteratively adjusts the trendline slope to optimize intersection with support/resistance points.
    - `get_optimized_trendlines`: Combines the above to return both support and resistance trendlines for a given range.
- The Dash app layout includes:
    - A candlestick chart, RSI chart, and MACD chart.
    - Controls for trendline display and date range selection.
- Dash callbacks handle:
    - Updating date labels based on slider position.
    - Updating all charts when the date range or trendline visibility changes, including recalculating and plotting trendlines.
- The app runs in debug mode for interactive development.
Usage:
------
Run the script. Open the provided local URL (localhost:8050 by default) in a browser to interact with the dashboard.
Select date ranges and toggle trendlines to analyze stock price action and technical levels.
Dependencies:
-------------
- yfinance
- dash
- plotly
- pandas
- numpy
- scipy
"""

import yfinance as yf
import dash
from dash import dcc, html
import pandas as pd
import plotly.graph_objs as go
import numpy as np


from dash.dependencies import Input, Output, State
from scipy.signal import argrelextrema

#======================================================
# Calculate the Relative Strength Index (RSI) for a given pandas Series.
def calculate_rsi(series, period=14):
    """
    Calculate the Relative Strength Index (RSI) for a given pandas Series.
    :param series: pandas Series of prices
    :param period: Number of periods to use for RSI calculation (default is 14)
    :return: pandas Series of RSI values
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

#======================================================
# Fast trendline calculation
def fast_trendline(hist, start_index, end_index):
    """
    Calculate a fast linear trendline for the 'Close' price between start_index and end_index (inclusive).
    Returns x (index/dates) and y (trendline values).
    """
    sub_hist = hist[start_index:end_index]
    if len(sub_hist) < 2:
        return [], []
    x = np.arange(len(sub_hist))
    y = sub_hist['Close'].values
    coef = np.polyfit(x, y, 1)
    trend_y = coef[0] * x + coef[1]
    return list(sub_hist.index), list(trend_y)


#======================================================
# Find the most extreme point (support or resistance) in the data
# Example usage:
# support_point = find_extreme_point('support', hist.tail(30), np.array(y_trend))
# resistance_point = find_extreme_point('resistance', hist.tail(30), np.array(y_trend))
def find_extreme_point(type, input_data, trendline):
    """
    Finds the most extreme point (support or resistance) in the input_data based on the trendline.
    For 'support', it finds the lowest point in input_data that is closest to the trendline.
    For 'resistance', it finds the highest point in input_data that is closest to the trendline.
    Returns the index, value, and distance of the extreme point.
    """
    if type == 'support':
        diffs = trendline - input_data['Low'].values
        idx = np.argmax(diffs)
        return input_data.index[idx], input_data['Low'].iloc[idx], diffs[idx]
    elif type == 'resistance':
        diffs = input_data['High'].values - trendline
        idx = np.argmax(diffs)
        return input_data.index[idx], input_data['High'].iloc[idx], diffs[idx]
    else:
        return None
    
#======================================================
# Count how many chart points are intersected by the trendline.
# This function is used to count the number of intersections between the trendline and the chart points.
def count_trendline_intersections(trendline, input_data, precision, type):
    """
    Count how many chart points (High or Low) are intersected by the trendline.
    :param trendline: The trendline values (numpy array or list).
    :param input_data: The input data containing 'High' or 'Low' values.
    :param precision: The precision for counting intersections.
    :param type: 'support' or 'resistance' to determine which values to compare against.
    :return: The count of intersections.
    """  
    count = 0
    if type == 'support':
        values = input_data['Low'].values
    elif type == 'resistance':
        values = input_data['High'].values
    else:
        return 0

    for i in range(len(trendline)):
        if abs(trendline[i] - values[i]) <= precision:
            count += 1
    return count

#======================================================
# Optimize the trendline intersections
def optimize_trendline_intersections(type, slope, extreme_point_idx, start_index, end_index, input_data, precision=0.000001, max_iter=100):
    """
    Iteratively adjust the slope of a trendline through the extreme point (by index) to minimize the distance
    to the closest Low (for support) or High (for resistance) value.
    param type: 'support' or 'resistance'.
    param slope: Initial slope of the trendline.
    param extreme_point_idx: Index of the extreme point in the input_data.
    param start_index: Start index for the sub-histogram.
    param end_index: End index for the sub-histogram.
    param input_data: The input data containing 'High', 'Low', or 'Close' values.
    param precision: Precision for the distance calculation.
    param max_iter: Maximum number of iterations to adjust the slope.
    return: Tuple of the optimized trendline (numpy array) and the date index of the closest point.
    """
    sub_hist = input_data[start_index:end_index]
    n = len(sub_hist)
    best_slope = slope
    delta_slope = 0.01 * abs(slope) if slope != 0 else 0.01

    # Get the y-value of the extreme point
    if type == 'support':
        ref_value = input_data['Low'][extreme_point_idx]
        values = sub_hist['Low'].values
    elif type == 'resistance':
        ref_value = input_data['High'][extreme_point_idx]
        values = sub_hist['High'].values
    else:
        ref_value = input_data['Close'][extreme_point_idx]
        values = sub_hist['Close'].values

    # Convert extreme_point_idx (timestamp) to integer position in sub_hist
    idx_pos = sub_hist.index.get_loc(extreme_point_idx)

    prev_distance = float('inf')
    direction = 1  # 1 for increasing slope, -1 for decreasing

    for iter_count in range(max_iter):
        intercept = ref_value - best_slope * idx_pos
        trendline = best_slope * np.arange(n) + intercept

        distances = np.abs(trendline - values)
        distances[idx_pos] = np.inf  # Ignore the extreme point itself
        min_distance = np.min(distances)
        closest_idx = np.argmin(distances)

        if min_distance <= precision:
            date_idx = sub_hist.index[closest_idx]
            return trendline, date_idx

        if min_distance > prev_distance:
            direction *= -1
            delta_slope *= 0.5

        if type == 'resistance':
            best_slope += direction * delta_slope
        elif type == 'support':
            best_slope -= direction * delta_slope

        prev_distance = min_distance

    date_idx = sub_hist.index[closest_idx]
    return trendline, date_idx

#======================================================
# Get optimized support and resistance trendlines
def get_optimized_trendlines(start_index, end_index, precision=0.000001, input_data=None):
    """
    Calculate optimized support and resistance trendlines for the data between start_index and end_index.
    param start_index: Start index for the sub-histogram.
    param end_index: End index for the sub-histogram.
    param precision: Precision for the distance calculation.
    param input_data: The input data containing 'High', 'Low', or 'Close' values.
    :return: Dictionary with 'support' and 'resistance' trendlines.
    """
    x_trend, y_trend = fast_trendline(input_data, start_index, end_index)
    if not x_trend or not y_trend:
        return {'support': None, 'resistance': None}

    y_trend_arr = np.array(y_trend)
    sub_hist = input_data[start_index:end_index]

    # Find extreme points
    support_idx, support_val, _ = find_extreme_point('support', sub_hist, y_trend_arr)
    resistance_idx, resistance_val, _ = find_extreme_point('resistance', sub_hist, y_trend_arr)

    # Initial slope from fast_trendline
    n = len(sub_hist)
    if n > 1:
        initial_slope = (y_trend[-1] - y_trend[0]) / (n - 1)
    else:
        initial_slope = 0.0

    # Optimize support trendline using the support extreme point index
    support_trendline, support_idx_2nd_point = optimize_trendline_intersections(
        'support', initial_slope, support_idx, start_index, end_index, input_data, precision=precision, max_iter=500
    )

    # Optimize resistance trendline using the resistance extreme point index
    resistance_trendline, resistance_idx_2nd_point = optimize_trendline_intersections(
        'resistance', initial_slope, resistance_idx, start_index, end_index, input_data, precision=precision, max_iter=500
    )

    return {
        'support': support_trendline,
        'resistance': resistance_trendline
    }


#======================================================

stock_name = "OTP.BD"  # Example stock ticker
#stock_name = "AAPL"  # Example: Apple Inc.
#stock_name = "GOOGL"  # Example: Alphabet Inc. (Google)
#stock_name = "MSFT"  # Example: Microsoft Corporation
#stock_name = "TSLA"  # Example: Tesla Inc.
#stock_name = "BTC-USD"  # Example: Amazon.com Inc.


# Fetch historical data for Apple Inc. (AAPL)
ticker = yf.Ticker(stock_name)
hist = ticker.history(period="1y")  # 1 year of historical data


# Calculate RSI
hist['RSI'] = calculate_rsi(hist['Close'])

# Calculate MACD
exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
hist['MACD'] = exp1 - exp2
hist['Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()

app = dash.Dash(__name__)

# ========================= Create Stock Price Candlestick Chart ============
candlestick_fig = go.Figure()
candlestick_fig.add_trace(go.Candlestick(
    x=hist.index,
    open=hist['Open'],
    high=hist['High'],
    low=hist['Low'],
    close=hist['Close'],
    name='Candlestick'
))
candlestick_fig.update_layout(
    title=f"{stock_name} Stock Price - Last 1 Year",
    xaxis_title='Date',
    yaxis_title='Price',
    legend=dict(
        orientation="h",
        x=0.5,
        y=1.2,
        xanchor='center',
        yanchor='top'
    )
)

# =============================== Create RSI figure ========================
rsi_fig = go.Figure()
rsi_fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], mode='lines', name='RSI'))
rsi_fig.update_layout(title='RSI (14)', xaxis_title='Date', yaxis_title='RSI')

# =============================== Create MACD figure =======================
macd_fig = go.Figure()
macd_fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], mode='lines', name='MACD'))
macd_fig.add_trace(go.Scatter(x=hist.index, y=hist['Signal'], mode='lines', name='Signal'))
macd_fig.update_layout(title='MACD', xaxis_title='Date', yaxis_title='Value')

# ======================= Create a Dash application layout =================
app.layout = html.Div([
    html.H1("OTP.BD Stock Price - Last 1 Year"),
    dcc.Store(id='trendline-points', data=[]),  # Store for clicked points
    dcc.Checklist(
        id='show-trendline',
        options=[{'label': 'Show Trendline', 'value': 'show'}],
        value=[],
        style={'margin': '10px'}
    ),
    dcc.Graph(
        id='stock-chart',
        figure= candlestick_fig,
    ),
    dcc.RangeSlider(
        id='date-range-slider',
        min=0,
        max=len(hist.index) - 1,
        value=[0, len(hist.index) - 1],
        marks={i: hist.index[i].strftime('%Y-%m-%d') for i in range(0, len(hist.index), max(1, len(hist.index)//10))},
        step=1,
        allowCross=False,
        tooltip={"placement": "bottom", "always_visible": True},
        updatemode='mouseup',
    ),
    html.Div(id='slider-date-label', style={'margin': '10px', 'fontWeight': 'bold'}),
    dcc.Graph(
        id='rsi-chart',
        figure=rsi_fig
    ),
    dcc.Graph(
        id='macd-chart',
        figure=macd_fig
    )
])

# ================ Callback to update the date range slider value with date labels instead of index ===========
@app.callback(
    Output('slider-date-label', 'children'),
    Input('date-range-slider', 'value')
)
def update_slider_date_label(value):
    start_idx, end_idx = value
    start_date = hist.index[start_idx].strftime('%Y-%m-%d')
    end_date = hist.index[end_idx].strftime('%Y-%m-%d')
    return f"Selected range: {start_date} to {end_date}"


# ================================ Callback to update the charts ======================
@app.callback(
    Output('stock-chart', 'figure'),
    Output('rsi-chart', 'figure'),
    Output('macd-chart', 'figure'),
    Input('trendline-points', 'data'),
    Input('show-trendline', 'value'),
    Input('date-range-slider', 'value')
)
def update_chart(points, show_trendline, slider_value):
    start_idx, end_idx = slider_value
    start_date = hist.index[start_idx]
    end_date = hist.index[end_idx]

    cp_rsi_fig = go.Figure(rsi_fig)  # Create a copy to avoid modifying the original

    # Update RSI chart with the selected date range lines
    cp_rsi_fig.update_layout(shapes=[
        dict(
            type="line",
            xref="x",
            yref="paper",
            x0=start_date,
            x1=start_date,
            y0=0,
            y1=1,
            line=dict(color="blue", width=1, dash="dot"),
        ),
        dict(
            type="line",
            xref="x",
            yref="paper",
            x0=end_date,
            x1=end_date,
            y0=0,
            y1=1,
            line=dict(color="blue", width=1, dash="dot"),
        ),
    ])

    # Update MACD chart with the selected date range lines
    cp_macd_fig = go.Figure(macd_fig)  # Create a copy to avoid modifying the original

    cp_macd_fig.update_layout(shapes=[
        dict(
            type="line",
            xref="x",
            yref="paper",
            x0=start_date,
            x1=start_date,
            y0=0,
            y1=1,
            line=dict(color="blue", width=1, dash="dot"),
        ),
        dict(
            type="line",
            xref="x",
            yref="paper",
            x0=end_date,
            x1=end_date,
            y0=0,
            y1=1,
            line=dict(color="blue", width=1, dash="dot"),
        ),
    ])
    
    #update candlestick chart with the selected date range lines
    cp_stock_fig = go.Figure(candlestick_fig)  # Create a copy to avoid modifying the original
    cp_stock_fig.update_layout(
        shapes=[
            dict(
                type="line",
                xref="x",
                yref="paper",
                x0=start_date,
                x1=start_date,
                y0=0,
                y1=1,
                line=dict(color="blue", width=1, dash="dot"),
            ),
            dict(
                type="line",
                xref="x",
                yref="paper",
                x0=end_date,
                x1=end_date,
                y0=0,
                y1=1,
                line=dict(color="blue", width=1, dash="dot"),
            ),
        ]
    )

    # Calculate and show the trendline if the checkbox is selected
    if show_trendline == ['show']:
        # Calculate and plot the fast trendline
        x_trend, y_trend = fast_trendline(hist, start_idx, end_idx)
        if x_trend and y_trend:
            cp_stock_fig.add_trace(go.Scatter(
            x=x_trend,
            y=y_trend,
            mode='lines',
            name='Fast Trendline',
            line=dict(color='orange', width=1)
            ))

        # Calculate and plot the optimized support
        trendlines = get_optimized_trendlines(start_idx,end_idx , precision=0.1, input_data=hist)
        if trendlines['support'] is not None:
            support_trendline = trendlines['support']
            x_trend = hist.index[start_idx:end_idx + 1]
            cp_stock_fig.add_trace(go.Scatter(
                x=x_trend,
                y=support_trendline,
                mode='lines',
                name='Support Trendline',
                line=dict(color='green', width=1)
            ))
        # Calculate and plot the resistance
        if trendlines['resistance'] is not None:
            resistance_trendline = trendlines['resistance']
            x_trend = hist.index[start_idx:end_idx + 1]
            cp_stock_fig.add_trace(go.Scatter(
                x=x_trend,
                y=resistance_trendline,
                mode='lines',
                name='Resistance Trendline',
                line=dict(color='red', width=1)
            ))


        ######################## Prolong the trendlines over the next 20 days after the end date ####################
        # 1. Calculate the new end index (prolonged, but not past last date)
        prolonged_end_idx = min(end_idx + 20, len(hist.index) - 1)

        # 2. Get the x-axis values for the prolonged range
        x_trend_prolonged = hist.index[start_idx:prolonged_end_idx + 1]

        # 3. For plotting, extend the trendline y-values accordingly
        # If your trendline is calculated as y = slope * x + intercept, recalculate for the extended x
        n_prolonged = len(x_trend_prolonged)
        if trendlines['support'] is not None:
            # Recalculate the trendline for the extended range
            # Use the same slope and intercept as before
            support_slope = (trendlines['support'][-1] - trendlines['support'][0]) / (end_idx - start_idx) if end_idx > start_idx else 0
            support_intercept = trendlines['support'][0]
            support_trendline_prolonged = [support_slope * i + support_intercept for i in range(n_prolonged)]
            cp_stock_fig.add_trace(go.Scatter(
                x=x_trend_prolonged,
                y=support_trendline_prolonged,
                mode='lines',
                name='Support Trendline (Prolonged)',
                line=dict(color='green', width=1, dash='dash')
            ))

        if trendlines['resistance'] is not None:
            resistance_slope = (trendlines['resistance'][-1] - trendlines['resistance'][0]) / (end_idx - start_idx) if end_idx > start_idx else 0
            resistance_intercept = trendlines['resistance'][0]
            resistance_trendline_prolonged = [resistance_slope * i + resistance_intercept for i in range(n_prolonged)]
            cp_stock_fig.add_trace(go.Scatter(
                x=x_trend_prolonged,
                y=resistance_trendline_prolonged,
                mode='lines',
                name='Resistance Trendline (Prolonged)',
                line=dict(color='red', width=1, dash='dash')
            ))
        ###########################################################################################

    return cp_stock_fig, cp_rsi_fig, cp_macd_fig


# ========================= Run the Dash application =========================
if __name__ == '__main__':
    app.run(debug=True)


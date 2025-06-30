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
from plotly.subplots import make_subplots
import numpy as np
import stock as st

import logging as logging


from dash.dependencies import Input, Output, State


# Create candlestick figure from input data and title parameters
def create_candlestick_figure(stock):
    """ Create a candlestick figure from input data and title parameters.
    Args:
        stock object: using get_historical_data to get pd.DataFrame with 'Open', 'High', 'Low', 'Close' columns.
        title (str): Title for the candlestick chart."""
    
    # Check if historical data is already fetched    
    if stock.get_historical_data().empty:
        # if not, fetch it
        stock._fetch_historical_data()  
    
    #Create candlestick
    candlesticks = go.Candlestick(
        x=stock.get_indexes(),
        open=stock.get_historical_data('Open'),
        high=stock.get_historical_data('High'),
        low=stock.get_historical_data('Low'),
        close=stock.get_historical_data('Close'),
        name='Candlestick'
    )

    #Create (filter out) separate data frames in order to be able to show red & green bars depending on Close/Open price relation
    green_volume_df = stock.get_historical_data()[stock.get_historical_data()['Close'] >= stock.get_historical_data()['Open']]
    red_volume_df = stock.get_historical_data()[stock.get_historical_data()['Close'] < stock.get_historical_data()['Open']]

    #Create green volume bars from green volume data frames
    volume_bars_green = go.Bar(
        x=green_volume_df.index,
        y=green_volume_df['Volume'],
        showlegend=False,
        marker={"color":"rgba(10,128,10,0.3)",},
        name='Volume',
        visible = False
    )

    #Create red volume bars from red volume data frames
    volume_bars_red = go.Bar(
        x=red_volume_df.index,
        y=red_volume_df['Volume'],
        showlegend=False,
        marker={"color":"rgba(128,10,10,0.3)",},
        name='Volume',
        visible = False
    )

    #Create CandleStick figure    
    figure = go.Figure(candlesticks)
    #Create sub-plots in order to show more different charts on the same graph -> enable secondary y axis
    figure = make_subplots(specs=[[{"secondary_y" : True}]]) 
    #Add candlestick trace to candlestick figure
    figure.add_trace(candlesticks,secondary_y=True)
    #Add green and red traces to candlestick figure
    figure.add_trace(volume_bars_green, secondary_y = False)
    figure.add_trace(volume_bars_red, secondary_y = False)

    #update figure layout
    figure.update_layout(
        title=f"{stock.name} Stock Price - Last {stock.period}",
        xaxis_title='Date',
        yaxis_title='Volume',
        legend=dict(
            orientation="h",
            x=0.5,
            y=1.35,
            xanchor='center',
            yanchor='top',
            font=dict(
                size=11,
            ),
        ),
    )


    #xaxes type set to category in order to avoid gaps on charts where there was no trade
    #Todo: Xaxes labales still have to adjusted because it is not handled automaticaly like if xaxes were date type
    figure.update_xaxes(type='category', row=1, col=1, dtick=60, tickformat= '%Y-%m-%d', tickangle=0)
    figure.update_xaxes(type='category', row=2, col=1, dtick=60, tickformat= '%Y-%m-%d', tickangle=0)


    # figure.update_xaxes(type='category', row=1, col=1, showticklabels=False)
    # figure.update_xaxes(type='category', row=2, col=1, showticklabels=False)

    ###figure.update_xaxes(type='category', row=1, col=1, dtick=7, tickmode='sync')
    ###figure.update_xaxes(type='category', row=2, col=1, dtick=7, tickmode='sync')

    # figure.update_xaxes(type='category', row=2, col=1,
    #                 tickformat= '%Y-%m-%d',
    #                 tickmode= 'array',
    #                 tickvals=hist.index,
    #                 ticktext=hist.index.strftime('%Y-%m-%d'),
    #                 tickangle=0,
    #                 tickfont= {
    #                     'size': 10
    #                 },
    #                 tickson= 'boundaries',
    #                 ticklen= 10,
    #                 tickwidth= 2,
    #                 tickcolor= '#000',
    #                 automargin= True,
    #                 side= 'bottom')
    

    #tickvalues = [idx if idx % max(1, len(input_data.index)//10) == 0 else None for idx in range(len(input_data.index))]
    #print(tickvalues)
    

    # figure.update_xaxes(type='category', row=1, col=1,
    #                 tickformat= '%Y-%m-%d',
    #                 tickmode= 'array',
    #                 #tickvals=hist.index,
    #                 #ticktext=hist.index.strftime('%Y-%m-%d'),
    #                 tickvals=tickvalues
    #                 ticktext=tickvalues.index.strftime('%Y-%m-%d'),
    #                 tickangle=0,
    #                 tickfont= {
    #                     'size': 10
    #                 },
    #                 tickson= 'boundaries',
    #                 ticklen= 10,
    #                 tickwidth= 2,
    #                 tickcolor= '#000',
    #                 automargin= True,
    #                 side= 'bottom')

    

    #figure.update_yaxes(title_text='Price', row=1, col=1)

    return figure

# Create a RSI figure from input data and title parameters
def create_rsi_figure(stock, title):
    """ Create a RSI figure from input data and title parameters.
    Args:
        stock object: using get_historical_data to get pd.DataFrame with 'RSI' columns.
        title (str): Title for the RSI chart."""

    # # Get historical data from stock object
    # historical_data = stock.get_historical_data()

    # if historical_data is None:
    #     raise ValueError("Stock historical data is empty.")

    if stock.get_indicator('RSI') is None:
        # if not, calculate it
        stock._calculate_rsi()

        
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=stock.get_indexes(), y=stock.get_indicator('RSI'), mode='lines', name='RSI'))
    figure.update_layout(title=title, xaxis_title='Date', yaxis_title='RSI')

    return figure

# Create a MACD figure from input data and title parameters
def create_macd_figure(stock, title):
    """ Create a MACD figure from input data and title parameters.
    Args:
        stock object: using get_historical_data to get pd.DataFrame with 'MACD', 'Signal' columns.
        title (str): Title for the MACD chart."""
    
    # if stock.get_ is None:
    #     raise ValueError("Stock historical data is empty.")

    if stock.get_indicator('MACD') is None or stock.get_indicator('Signal') is None:
        # if not, calculate it
        stock._calculate_macd()
    
    figure = go.Figure()

    figure.add_trace(go.Scatter(x=stock.get_indexes(), y=stock.get_indicator('MACD'), mode='lines', name='MACD'))
    figure.add_trace(go.Scatter(x=stock.get_indexes(), y=stock.get_indicator('Signal'), mode='lines', name='Signal'))
    figure.update_layout(title=title, xaxis_title='Date', yaxis_title='Value')

    return figure


def trace_visibility(figure, trace_name, visible):
    """ Set the visibility of a trace in the figure."""
    for trace in figure.data:
        if trace.name == trace_name:
            trace.visible = visible

#======================================================

top_50_SandP_stocks = {
    "NDA-FI.HE": "Norde Bank Abp",
    "OTP.BD": "OTP Bank Nyrt.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "AAPL": "Apple Inc.",
    "AMZN": "Amazon.com, Inc.",
    "GOOG": "Alphabet Inc. (Class C)",
    "GOOGL": "Alphabet Inc. (Class A)",
    "META": "Meta Platforms, Inc.",
    "AVGO": "Broadcom Inc.",
    "BRK.B": "Berkshire Hathaway Inc. (Class B)",
    "TSLA": "Tesla, Inc.",
    "WMT": "Walmart Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "LLY": "Eli Lilly and Company",
    "V": "Visa Inc.",
    "ORCL": "Oracle Corporation",
    "NFLX": "Netflix, Inc.",
    "MA": "Mastercard Incorporated",
    "XOM": "Exxon Mobil Corporation",
    "COST": "Costco Wholesale Corporation",
    "JNJ": "Johnson & Johnson",
    "PG": "The Procter & Gamble Company",
    "HD": "The Home Depot, Inc.",
    "ABBV": "AbbVie Inc.",
    "BAC": "Bank of America Corporation",
    "PLTR": "Palantir Technologies Inc.",
    "KO": "The Coca-Cola Company",
    "PM": "Philip Morris International Inc.",
    "UNH": "UnitedHealth Group Incorporated",
    "TMUS": "T-Mobile US, Inc.",
    "CVX": "Chevron Corporation",
    "CSCO": "Cisco Systems, Inc.",
    "GE": "General Electric Company",
    "IBM": "International Business Machines Corporation",
    "CRM": "Salesforce, Inc.",
    "ABT": "Abbott Laboratories",
    "WFC": "Wells Fargo & Company",
    "LIN": "Linde plc",
    "MCD": "McDonald's Corporation",
    "DIS": "The Walt Disney Company",
    "INTU": "Intuit Inc.",
    "MS": "Morgan Stanley",
    "MRK": "Merck & Co., Inc.",
    "NOW": "ServiceNow, Inc.",
    "T": "AT&T Inc.",
    "AXP": "American Express Company",
    "ACN": "Accenture plc",
    "RTX": "RTX Corporation",
    "AMD": "Advanced Micro Devices, Inc.",
    "ISRG": "Intuitive Surgical, Inc.",
    "VZ": "Verizon Communications Inc."
}

element_side_margins_1 = '65px'         # left and right margins for elements - #1
max_dropdown_width = '300px'  # max width of the dropdown menu
default_stock = 'NDA-FI.HE'  #Default stock name
current_stock = "NDA-FI.HE"  # Default selected stock ticker

previous_range_slider_value = None # Stores the range slider value from previous callback to avoid necessary updates
trendline_pronogation = 40 # Number of days to prolong trendlines beyond the end range marker

# Checkbox list for indicators
Indicator_list = ['Moving Average 1', 'Moving Average 2', 'Trendlines', 'Bollinger Bands', 'Volume']


# Initial 'Marker Mode' button style
#marker_mode_btn_style_default = {'background-color': 'white', 'color': 'black', 'height': '30px', 'width': '100px'}
marker_mode_btn_style_default = {'background-color': 'white'}
# Selected 'Marker Mode' button style
#marker_mode_btn_style_selected = {'background-color': 'orange', 'color': 'black', 'height': '30px', 'width': '100px'}
marker_mode_btn_style_selected = {'background-color': 'orange'}


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) #Set logger level to DEBUG

# ================ Fetch historical data and calculate indicators =================
# Create stock object, fetch historical data, and calculate indicators
stock = st.Stock(current_stock)  # Create an instance of the Stock class


# ========================= Create Stock Price Candlestick Chart ============
candlestick_fig = create_candlestick_figure(stock)

# =============================== Create RSI figure ========================
rsi_fig = create_rsi_figure(stock, 'RSI (14)')

# =============================== Create MACD figure =======================
macd_fig = create_macd_figure(stock, 'MACD (12, 26, 9)')


# ================ Create Dash application and layout =================
app = dash.Dash(__name__)

app.layout = html.Div([   
    html.Div([
        dcc.Dropdown(
            id='stock-dropdown',
            options=top_50_SandP_stocks,
            value=default_stock, # Default value set
            placeholder="Select a stock",
        ),
        ],
    style={'margin-left': element_side_margins_1, 'margin-right': element_side_margins_1, 'maxWidth': max_dropdown_width, 'margin-bottom': '20px'},
    ),
    html.Button('Marker Mode', id='marker-mode-button', n_clicks=0, style = marker_mode_btn_style_default),
    html.Div([
        dcc.Store(id='marker-positions', data={'marker1': None, 'marker2': None}),  # Store for marker clicked points
        dcc.Store(id='graph-mode', data={'marker_mode': None }),  # Store the current graph mode
        dcc.Checklist(Indicator_list, id='indicator-checklist', inline=True, inputStyle={'margin-right': '10px'}, labelStyle={'display': 'inline-block', 'margin-right': '20px'}, style={'margin-left': '80px'}),
        dcc.Graph(
            id='stock-chart',
            figure= candlestick_fig,
            config={'modeBarButtonsToAdd': ['drawline']},
        ),
    ]),
    dcc.Graph(
        id='rsi-chart',
        figure=rsi_fig
    ),
    dcc.Graph(
        id='macd-chart',
        figure=macd_fig
    )
])

#==================================  Chart Markers Handling ============================

# Add markers to figure
def add_markers_to_figure(fig, marker_positions):
    if fig != None:
        if marker_positions['marker1'] is not None:
            logger.debug(f"Draw marker line 1 to {fig['layout']['title']['text']} position: {marker_positions['marker1']}")
            fig.add_vline(x=marker_positions['marker1'], name='marker1', line_width=1, line_dash="dot", line_color="grey")

    # If Marker 2 position is available add vertical line to Marker 2 position
    if marker_positions['marker2'] is not None:
            logger.debug(f"Draw line 2 to {fig['layout']['title']['text']} position: {marker_positions['marker2']}")
            fig.add_vline(x=marker_positions['marker2'], name='marker2', line_width=1, line_dash="dot", line_color="grey")

#Remove markers (shapes in names 'marker) from figure
def remove_markers_from_figure(fig):
    #Remove marker shapes which already added earlier
    updated_shapes = []
    for shape in fig['layout']['shapes']:
        #Serach for shapes which has no 'marker' in their names
        if 'marker' not in shape['name']:
            #add them to a new shape list
            updated_shapes.append(shape)
    #update the figure shape list with the new one filtered out the markers
    fig['layout']['shapes'] = updated_shapes

# Callback to toggle graph marker mode and change marker mode button style depending on marker mode
@app.callback(
    Output('graph-mode', 'data', allow_duplicate=True),
    Output('marker-mode-button', 'style'),
    Input('marker-mode-button', 'n_clicks'),
    State('graph-mode', 'data'),
    prevent_initial_call=True
)

def update_graph_mode(n_clicks, graph_mode):
    """ Callback function to toggle graph marker mode on/off
    :param n_click: number of clicks on button 
    :param graph_mode: the current graph-mode 
    return changed graph mode 
    """
    # If Marker mode is None switch it to Marker1 mode
    if graph_mode['marker_mode'] is None:
        graph_mode['marker_mode'] = 'Marker1'
        btn_style = marker_mode_btn_style_selected
    else:
        graph_mode['marker_mode'] = None
        btn_style = marker_mode_btn_style_default

    return graph_mode, btn_style

#Callback to update marker lines on stock-chart
@app.callback(
    Output('stock-chart', 'figure', allow_duplicate=True),
    Output('rsi-chart', 'figure', allow_duplicate=True),
    Output('macd-chart', 'figure', allow_duplicate=True),
    Output('marker-positions', 'data'),
    Output('graph-mode', 'data', allow_duplicate=True),
    Input('stock-chart', 'clickData'),
    State('marker-positions', 'data'),
    State('graph-mode', 'data'),
    prevent_initial_call=True
)
def draw_vertical_line(clickData, marker_positions, graph_mode):
    """ Callback to update marker positions on stock chart
    Functions updates Marker1 and Marker2 positions in dcc.Store(marker-position) and draws these two markers to the chart.
    On each click on charts it toggles marker mode between Marker1 & Marker 2 to update the corresponding marker line

    :param clickData: data get from mouse click on chart
    :param marker_position: dictionary stores marker 1 & marker 2 postitions
    :param graph_mode: dictionarty stores the grap-mode
    return: new graph figure, updated marker positions and graph mode
    """
    #If callback called not because click on chart or the Marker Mode is off, simply return without doing anything
    if clickData is None or graph_mode['marker_mode'] == None:
        return dash.no_update, dash.no_update, dash.no_update, marker_positions, graph_mode

    #Gets the date to the mouse click position
    date_clicked = clickData['points'][0]['x']

    
    if graph_mode['marker_mode'] == 'Marker1':
        #if current Marker mode is Marker1 -> Update Marker 1 position and change the mode to Marker 2
        marker_positions['marker1'] = date_clicked
        graph_mode['marker_mode'] = 'Marker2'
    else:
        #if current Marker mode is Marker2 -> Update Marker 2 position and change the mode to Marker 1
        marker_positions['marker2'] = date_clicked
        graph_mode['marker_mode'] = 'Marker1'

    # Make a copy of figures
    fig_stock = go.Figure(candlestick_fig)
    fig_rsi = go.Figure(rsi_fig)
    fig_macd = go.Figure(macd_fig)

    #Remove markers from figure if there is any
    remove_markers_from_figure(fig_stock)

    #Add markers to charts
    add_markers_to_figure(fig_stock, marker_positions)
    add_markers_to_figure(fig_rsi, marker_positions)
    add_markers_to_figure(fig_macd, marker_positions)

    return fig_stock, fig_rsi, fig_macd, marker_positions, graph_mode

#===================================================================================

@app.callback(
    Output('stock-chart', 'figure', allow_duplicate= True),
    Output('rsi-chart', 'figure', allow_duplicate= True),
    Output('macd-chart', 'figure', allow_duplicate= True),
    Output('marker-positions', 'data', allow_duplicate= True),
    Output('indicator-checklist', 'value'),
    Input('stock-dropdown', 'value'),
    State('marker-positions', 'data'),
    State('indicator-checklist', 'value'),
    prevent_initial_call=True
)
def update_chart(stock_item, marker_positions, indicator_checklist):
    global current_stock, candlestick_fig, rsi_fig, macd_fig, stock, previous_range_slider_value, trendline_pronogation

    if stock_item != current_stock:
        logger.debug(f"Create new stock object for {stock_item} for period {stock_item}")
        # Create new stock object for the selected stock
        stock = st.Stock(stock_item)
        current_stock = stock_item  # Update the selected stock

        # Recreate Candlestick, RSI, MACD figures with new data
        logger.debug(f"Create new Candlestick, MACD, RSI figures for {stock_item}")
        cp_stock_fig = create_candlestick_figure(stock)
        cp_rsi_fig = create_rsi_figure(stock, 'RSI (14)')
        cp_macd_fig = create_macd_figure(stock, 'MACD (12, 26, 9)') 

        candlestick_fig = cp_stock_fig
        rsi_fig = cp_rsi_fig
        macd_fig = cp_macd_fig

        #For new figure delete the markers
        marker_positions['marker1'] = None
        marker_positions['marker2'] = None

    return candlestick_fig, rsi_fig, macd_fig, marker_positions, indicator_checklist


# ================================ Callback to update the charts ======================
@app.callback(
    Output('stock-chart', 'figure', allow_duplicate= True),
    Input('indicator-checklist', 'value'),
    State('marker-positions', 'data'),
    prevent_initial_call=True
)
def update_chart(indicators_selected, marker_positions):

    global current_stock, candlestick_fig, rsi_fig, macd_fig, stock, previous_range_slider_value, trendline_pronogation

    update_trendlines = False #Flag to indicate if trendlines data needs to be recalculated and added as a new trace to chart

    # If the stock is the same, use the existing figures
    cp_stock_fig = go.Figure(candlestick_fig)  # Create a copy to avoid modifying the original
    cp_rsi_fig = go.Figure(rsi_fig) # Create a copy to avoid modifying the original
    cp_macd_fig = go.Figure(macd_fig) # Create a copy to avoid modifying the original
 
    if marker_positions['marker1'] != None and marker_positions['marker2'] != None:
        if marker_positions['marker1'] < marker_positions['marker2']:
            start_date = marker_positions['marker1']
            end_date = marker_positions['marker2']
        else:
            start_date = marker_positions['marker2']
            end_date = marker_positions['marker1']
        
        start_idx = stock.historical_data.index.get_loc(start_date)
        end_idx = stock.historical_data.index.get_loc(end_date)
    else:
        start_idx = 0
        end_idx = stock.get_number_of_idx() - 1   
        start_date = stock.get_indexes()[start_idx]
        end_date = stock.get_indexes()[end_idx]

    # Remove the vertical line (all shapes from figure)
    remove_markers_from_figure(cp_stock_fig)
    # Add markers to stock figure
    add_markers_to_figure(cp_stock_fig, marker_positions)

    logger.debug(f"Indicators selected: {indicators_selected}")

    # Create a list of the chart trace names in candlestick figure to use it later if trace is already added to the figure
    traces_names_in_stock_figure = [trace.name for trace in cp_stock_fig.data]
    
    if indicators_selected != None:
        #Calculate and plot the first moving average if selected
        if 'Moving Average 1' in indicators_selected:
            # Check if Moving Average was already calculated
            if stock.get_indicator('Moving Average 1') is None:
                # if not, calculate it
                stock._calculate_moving_averages('Moving Average 1', 20)

            # If 'Moving Average 1 (20-day)' trace already exists, set it to visible
            if 'Moving Average 1 (20-day)' in traces_names_in_stock_figure:
                    trace_visibility(cp_stock_fig, 'Moving Average 1 (20-day)', True)
            else:
                cp_stock_fig.add_trace(go.Scatter(
                    x=stock.get_indexes(),
                    y=stock.get_indicator('Moving Average 1'),
                    mode='lines',
                    name='Moving Average 1 (20-day)',
                    line=dict(color='blue', width=1),
                ),secondary_y=True,),
        else:
            # Set trace visibility to False
            trace_visibility(cp_stock_fig, 'Moving Average 1 (20-day)', False)

        # Calculate and plot the second moving average if selected
        if 'Moving Average 2' in indicators_selected:
            # Check if Moving Average was already calculated
            if stock.get_indicator('Moving Average 2') is None:
                # if not, calculate it
                stock._calculate_moving_averages('Moving Average 2', 50)

            # If 'Moving Average 2 (50-day)' trace already exists, set it to visible
            if 'Moving Average 2 (50-day)' in traces_names_in_stock_figure:
                trace_visibility(cp_stock_fig, 'Moving Average 2 (50-day)', True)
            else:
                # Otherwise add a new trace for Moving Average 2
                cp_stock_fig.add_trace(go.Scatter(
                    x=stock.get_indexes(),
                    y=stock.get_indicator('Moving Average 2'),
                    mode='lines',
                    name='Moving Average 2 (50-day)',
                    line=dict(color='red', width=1)
                ),secondary_y=True,)
        else:
            # Set trace visibility to False
            trace_visibility(cp_stock_fig, 'Moving Average 2 (50-day)', False)

        # Calculate and show the trendline if the checkbox is selected
        # or just recalulate and replace traces in chart figure when range slider value changed
        if ('Trendlines' in indicators_selected) or (update_trendlines == True):
            # Calculate trendline if it hasn't been calculated yet, or needs to be recalculated due to range value change
            if (stock.trendline is None) or (update_trendlines == True):
                # if not, calculate it
                stock._calculate_trendline(start_idx, end_idx,prolongation=trendline_pronogation)
                # Calculate Support and Resistance trendlines 
                stock._calculate_sup_res_trendlines(start_idx, end_idx, 0.000001,prolongation=trendline_pronogation)

        #If trendlines needs to be upadted replace trendline data in the figure traces
        if update_trendlines == True:
            print("replace tradline traces")
            for trace in cp_stock_fig.data:
                if trace.name == 'Trendline':
                    trace.x=stock.trendline['index']
                    trace.y=stock.trendline['Trendline']              
                elif trace.name == 'Support Trendline':
                    trace.x=stock.trendline['index']
                    trace.y=stock.trendline['Support Trendline']
                else:
                    if trace.name == 'Resistance Trendline':
                        trace.x=stock.trendline['index']
                        trace.y=stock.trendline['Resistance Trendline']


        if ('Trendlines' in indicators_selected):
            if 'Trendline' in traces_names_in_stock_figure:
                trace_visibility(cp_stock_fig, 'Trendline', True)
                trace_visibility(cp_stock_fig, 'Support Trendline', True)
                trace_visibility(cp_stock_fig, 'Resistance Trendline', True)
            else:
                #Otherwise add the trendline as new traces
                cp_stock_fig.add_trace(go.Scatter(
                x=stock.trendline['index'],
                y=stock.trendline['Trendline'],
                mode='lines',
                name='Trendline',
                line=dict(color='orange', width=1)
                ),secondary_y=True,)
                #Add the support trendline as new traces
                cp_stock_fig.add_trace(go.Scatter(
                    x=stock.trendline['index'],
                    y=stock.trendline['Support Trendline'],
                    mode='lines',
                    name='Support Trendline',
                    line=dict(color='green', width=1)
                ),secondary_y=True,)
                #Add the resistance trendline as new traces
                cp_stock_fig.add_trace(go.Scatter(
                    x=stock.trendline['index'],
                    y=stock.trendline['Resistance Trendline'],
                    mode='lines',
                    name='Resistance Trendline',
                    line=dict(color='red', width=1)
                ),secondary_y=True,)

        else:
            trace_visibility(cp_stock_fig, 'Trendline', False)
            trace_visibility(cp_stock_fig, 'Support Trendline', False)
            trace_visibility(cp_stock_fig, 'Resistance Trendline', False)


        # Calculate and plot Bollinger Bands if selected
        if 'Bollinger Bands' in indicators_selected:
            # Chack if Bollinger Bands were already calculated
            if stock.get_indicator('Middle Bollinger Band') is None:
                # if not, calculate it
                stock._calculate_bollinger_bands(20)

            # If 'Bollinger' trace already exists, set it to visible
            if any('Bollinger Band' in trace for trace in traces_names_in_stock_figure):

                trace_visibility(cp_stock_fig, 'Upper Bollinger Band', True)
                trace_visibility(cp_stock_fig, 'Middle Bollinger Band', True)
                trace_visibility(cp_stock_fig, 'Lower Bollinger Band', True)
            else:
                cp_stock_fig.add_trace(go.Scatter(
                    x=stock.get_indexes(),
                    y=stock.get_indicator('Upper Bollinger Band'),
                    mode='lines',
                    name='Upper Bollinger Band',
                    line=dict(color='purple', width=1, dash='dash')
                ),secondary_y=True,)
                cp_stock_fig.add_trace(go.Scatter(
                    x=stock.get_indexes(),
                    y=stock.get_indicator('Middle Bollinger Band'),
                    mode='lines',
                    name='Middle Bollinger Band',
                    line=dict(color='orange', width=1, dash='dash')
                ),secondary_y=True,)
                cp_stock_fig.add_trace(go.Scatter(
                    x=stock.get_indexes(),
                    y=stock.get_indicator('Lower Bollinger Band'),
                    mode='lines',
                    name='Lower Bollinger Band',
                    line=dict(color='purple', width=1, dash='dash')
                ),secondary_y=True,)
        else:
            # Set Boillimger Band visibility to False
            trace_visibility(cp_stock_fig, 'Upper Bollinger Band', False)
            trace_visibility(cp_stock_fig, 'Middle Bollinger Band', False)
            trace_visibility(cp_stock_fig, 'Lower Bollinger Band', False)

        if 'Volume' in indicators_selected:
            # Set trace visibility to true
            if 'Volume' in traces_names_in_stock_figure:
                trace_visibility(cp_stock_fig, 'Volume', True)
        else:
            # Set trace visibility to false
            if 'Volume' in traces_names_in_stock_figure:
                trace_visibility(cp_stock_fig, 'Volume', False)

    candlestick_fig = cp_stock_fig
    rsi_fig = cp_rsi_fig
    macd_fig = cp_macd_fig

    return cp_stock_fig #, cp_rsi_fig, cp_macd_fig,



# ========================= Run the Dash application =========================
if __name__ == '__main__':
    app.run(debug=True)


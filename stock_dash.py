import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px

def calculate_metrics(data):
    price_change = data['Close'].iloc[-1] - data['Close'].iloc[0]
    total_return = data['Close'].pct_change().sum()
    dividend_yield = data['Dividends'].sum() / len(data)
    return price_change, total_return, dividend_yield

def daily_moves_histogram(data, bins, title):
    daily_moves = data['Close'].pct_change() * 100
    plt.hist(daily_moves, bins=bins, edgecolor='black')
    plt.axvline(x=1, color='red', linestyle='--')
    plt.axvline(x=-1, color='red', linestyle='--')
    plt.xlabel('Daily Moves (%)')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()
    
def hypothetical_growth(data, initial_investment=10000):
    data['hypothetical_growth'] = initial_investment * (1 + data['Close'].pct_change()).cumprod()
    return data['hypothetical_growth']

def plot_hypothetical_growth(data, title):
    data['hypothetical_growth'].plot()
    plt.xlabel('Date')
    plt.ylabel('Hypothetical Growth of $10,000')
    plt.title(title)
    plt.show()
    
def hypothetical_growth(data, initial_investment=10000, tax_rate=0.22):
    data['returns'] = data['Close'].pct_change()
    data['dividend_after_tax'] = data['Dividends'] * (1 - tax_rate)
    data['total_returns'] = (1 + data['returns']) * (1 + data['dividend_after_tax'] / data['Close'])
    data['hypothetical_growth'] = initial_investment * data['total_returns'].cumprod()
    return data['hypothetical_growth']

ticker = "AAPL" # Replace with desired stock ticker
stock = yf.Ticker(ticker)
ytd_data = stock.history(period="YTD")
three_year_data = stock.history(period="3y")
ten_year_data = stock.history(period="10y")

# Calculate listed items
ytd_metrics = calculate_metrics(ytd_data)
three_year_metrics = calculate_metrics(three_year_data)
ten_year_metrics = calculate_metrics(ten_year_data)

# Plot daily distribution histogram
#daily_moves_histogram(ten_year_data, bins=np.arange(-10, 10, 2), title='Daily Moves Distribution')

# Hypothetical growth
ytd_growth = hypothetical_growth(ytd_data)
three_year_growth = hypothetical_growth(three_year_data)
ten_year_growth = hypothetical_growth(ten_year_data)

#plot_hypothetical_growth(ten_year_data, 'Hypothetical Growth of $10,000 in 10 Years')

# Hypothetical growth with DRIP
ytd_growth_div_reinvest = hypothetical_growth(ytd_data)
three_year_growth_div_reinvest = hypothetical_growth(three_year_data)
ten_year_growth_div_reinvest = hypothetical_growth(ten_year_data)

#plot_hypothetical_growth(ten_year_data, 'Hypothetical Growth of $10,000 in 10 Years with Dividend Reinvestment')

# Create interactive dashboard
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.Label('Timeframe'),
            dcc.Dropdown(
                id='timeframe-dropdown',
                options=[
                    {'label': 'YTD', 'value': 'YTD'},
                    {'label': '3Y', 'value': '3Y'},
                    {'label': '10Y', 'value': '10Y'},
                ],
                value='YTD'
            ),
        ]),
    ]),
    html.Div(id='metrics'),
    dcc.Graph(id='daily-moves-histogram'),
    dcc.Graph(id='hypothetical-growth'),
    dcc.Graph(id='hypothetical-growth-drip')
])


@app.callback(
    [Output('metrics', 'children'),
     Output('daily-moves-histogram', 'figure'),
     Output('hypothetical-growth', 'figure'),
     Output('hypothetical-growth-drip', 'figure')],
    [Input('timeframe-dropdown', 'value')]
)
def update_dashboard(timeframe):
    if timeframe == 'YTD':
        data = ytd_data
        metrics = ytd_metrics
    elif timeframe == '3Y':
        data = three_year_data
        metrics = three_year_metrics
    elif timeframe == '10Y':
        data = ten_year_data
        metrics = ten_year_metrics

    metrics_div = html.Div([
        html.P(f"Price Change: {metrics[0]:.2f}"),
        html.P(f"Total Return: {metrics[1]:.2f}"),
        html.P(f"Dividend Yield: {metrics[2]:.2f}")
    ])

    daily_moves_histogram_fig = create_daily_moves_histogram_figure(data, bins=np.arange(-10, 10, 2))
    hypothetical_growth_fig = create_hypothetical_growth_figure(data)
    hypothetical_growth_drip_fig = create_hypothetical_growth_drip_figure(data)

    return metrics_div, daily_moves_histogram_fig, hypothetical_growth_fig, hypothetical_growth_drip_fig


def create_daily_moves_histogram_figure(data, bins):
    daily_moves = data['Close'].pct_change() * 100
    fig = px.histogram(daily_moves, nbins=len(bins), histnorm='percent', title='Daily Moves Distribution')
    fig.update_xaxes(title='Daily Moves (%)')
    fig.update_yaxes(title='Frequency (%)')
    return fig


def create_hypothetical_growth_figure(data):
    fig = px.line(data, x=data.index, y='hypothetical_growth', title='Hypothetical Growth of $10,000')
    fig.update_xaxes(title='Date')
    fig.update_yaxes(title='Hypothetical Growth of $10,000')
    return fig


def create_hypothetical_growth_drip_figure(data):
    fig = px.line(data, x=data.index, y='hypothetical_growth', title='Hypothetical Growth of $10,000 with Dividend Reinvestment')
    fig.update_xaxes(title='Date')
    fig.update_yaxes(title='Hypothetical Growth of $10,000 with Dividend Reinvestment')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
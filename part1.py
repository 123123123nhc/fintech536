from shiny import App, ui, render, reactive
import pandas as pd
import shinybroker as sb
import time
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns

tickers = ["QQQ", "XLV", "UUP", "TLT"]


def fetch_close_for_ticker(ticker):
    df = sb.fetch_historical_data(
        contract=sb.Contract({
            'symbol': ticker,
            'secType': "STK",
            'exchange': "SMART",
            'currency': "USD"
        }),
        barSizeSetting='1 day',
        durationStr='1 Y',
        whatToShow='ADJUSTED_LAST'
    )['hst_dta'][['timestamp', 'close']]
    df = df.rename(columns={'close': ticker})
    return df


# Fetch data once at app startup
historical_data = fetch_close_for_ticker(tickers[0])
for tk in tickers[1:]:
    historical_data = pd.merge(
        historical_data,
        fetch_close_for_ticker(tk),
        on='timestamp'
    )
    time.sleep(0.2)

historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
historical_data = historical_data.set_index('timestamp')
mu = expected_returns.mean_historical_return(historical_data)
S = risk_models.sample_cov(historical_data)


def run_optimization(risk_level):
    ef = EfficientFrontier(mu, S)
    if risk_level == 'low risk':
        weights = ef.min_volatility()
    elif risk_level == 'median risk':
        weights = ef.efficient_risk(target_volatility=0.15)
    elif risk_level == 'high risk':
        weights = ef.max_sharpe()
    return ef.clean_weights()


# App UI
app_ui = ui.page_fluid(
    ui.panel_title("Automated Portfolio Builder"),
    ui.input_select("risk", "Choose risk level:", choices=["low risk", "median risk", "high risk"],
                    selected="median risk"),
    ui.output_text("risk_output"),
    ui.output_table("weight_table")
)


# App logic
def server(input, output, session):
    @output
    @render.text
    def risk_output():
        return f"You selected: {input.risk()}"

    @output
    @render.table
    def weight_table():
        weights = run_optimization(input.risk())
        df = pd.DataFrame.from_dict(weights, orient="index", columns=["Weight (%)"])
        df["Weight (%)"] = (df["Weight (%)"] * 100).round(2)
        return df


app = App(app_ui, server)

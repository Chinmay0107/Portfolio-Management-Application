import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

# Page Configuration
st.set_page_config(page_title="Portfolio Management", layout="wide")


import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

# Function to calculate portfolio metrics
def calculate_portfolio_metrics(returns, weights):
    """
    Calculate portfolio metrics: return, volatility, and Sharpe ratio.
    Returns are annualized based on 252 trading days per year.
    """
    mean_daily_return = returns.mean()
    daily_covariance = returns.cov()

    # Annualized metrics
    annualized_return = np.dot(weights, mean_daily_return) * 252  # Scale daily return to annual
    annualized_volatility = np.sqrt(np.dot(weights.T, np.dot(daily_covariance, weights)) * 252)  # Annual volatility
    return annualized_return, annualized_volatility


def calculate_sharpe_ratio(annualized_return, annualized_volatility, risk_free_rate):
    """
    Calculate Sharpe Ratio with a user-defined risk-free rate.
    """
    return (annualized_return - risk_free_rate) / annualized_volatility


def calculate_max_drawdown(cumulative_returns):
    """
    Calculate Maximum Drawdown of the portfolio.
    """
    drawdowns = cumulative_returns / cumulative_returns.cummax() - 1
    return drawdowns.min()


def optimize_portfolio(returns):
    num_assets = len(returns.columns)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))

    # Define the objective function to maximize the Sharpe ratio
    def objective_function(weights):
        portfolio_return, portfolio_volatility = calculate_portfolio_metrics(returns, weights)
        sharpe_ratio = calculate_sharpe_ratio(portfolio_return, portfolio_volatility, risk_free_rate=0)
        return -sharpe_ratio  # Minimize the negative Sharpe ratio

    result = minimize(
        objective_function,  # Pass the objective function
        num_assets * [1. / num_assets],  # Equal weights as starting point
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result.x


# Welcome Page
st.title("📊 Portfolio Management Application")
st.markdown("""
**Welcome!** 
This interactive tool helps you create and optimize a portfolio or analyze an existing one.
""")
st.write("💡 _“Investing is about managing risk, not avoiding it.”_")

# Navigation Buttons
action = st.radio(
    "What would you like to do today?",
    ("Create an Optimal Portfolio", "Analyze My Existing Portfolio"),
    index=0
)

# --- Create an Optimal Portfolio ---
if action == "Create an Optimal Portfolio":
    st.subheader("Create an Optimal Portfolio")
    stocks = st.text_area(
        "Enter Stock Tickers (comma-separated, e.g., AAPL, MSFT, TSLA):", value="AAPL, MSFT, TSLA"
    )
    investment_amount = st.number_input("Enter Total Investment Amount ($):", min_value=100.0, value=1000.0, step=50.0)
    lookback_period = st.selectbox("Select Lookback Period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)

    risk_free_rate = st.number_input("Enter Risk-Free Rate (Annualized, %):", min_value=0.0, value=2.0) / 100

    if st.button("Run Optimization"):
        tickers = [ticker.strip().upper() for ticker in stocks.split(",")]
        try:
            # Fetch historical data and calculate daily returns
            historical_data = yf.download(tickers, period=lookback_period)["Adj Close"].dropna()
            daily_returns = historical_data.pct_change().dropna()

            # Calculate optimal weights
            optimal_weights = optimize_portfolio(daily_returns)

            # Calculate optimal portfolio metrics
            optimal_return, optimal_volatility = calculate_portfolio_metrics(daily_returns, optimal_weights)
            sharpe_ratio = calculate_sharpe_ratio(optimal_return, optimal_volatility, risk_free_rate)

            # Convert optimal return and volatility to percentages
            optimal_return_percentage = optimal_return * 100
            optimal_volatility_percentage = optimal_volatility * 100

            # Create allocation DataFrame
            optimal_allocation = pd.DataFrame({
                "Ticker": tickers,
                "Weight": optimal_weights,
                "Allocation ($)": optimal_weights * investment_amount
            })

            # Display Optimal Weights Table
            st.subheader("Optimal Portfolio Allocation")
            st.dataframe(optimal_allocation.style.format({"Weight": "{:.2%}", "Allocation ($)": "${:,.2f}"}))

            # Calculate Maximum Drawdown
            cumulative_returns = (1 + daily_returns.dot(optimal_weights)).cumprod()
            max_drawdown = calculate_max_drawdown(cumulative_returns)


            # Display Results
            def generate_card(title, value, description, color="black", bg_color="#f9f9f9", icon=None):
                icon_html = f"<img src='{icon}' alt='icon' style='width:24px; height:24px; margin-right:8px;'/>" if icon else ""
                return f"""
                <div style="
                    background-color: {bg_color};
                    color: black;
                    padding: 15px;
                    border-radius: 10px;
                    margin: 10px;
                    width: 220px;
                    font-family: Arial, sans-serif;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                    text-align: center;
                ">
                    <h3 style="margin: 0; color: black; font-size: 18px;">{icon_html}{title}</h3>
                    <p style="font-size: 22px; font-weight: bold; color: {color}; margin: 5px 0;">{value}</p>
                    <small style="color: gray; font-size: 14px;">{description}</small>
                </div>
                """


            # Create a grid-like layout for metrics
            st.markdown(
                """
                <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
                """ +
                generate_card(
                    "📊 Sharpe Ratio",
                    f"{sharpe_ratio:.2f}",
                    "Higher Sharpe Ratio indicates better risk-adjusted returns.",
                    color="green" if sharpe_ratio > 1 else "red"
                ) +
                generate_card(
                    "📉 Maximum Drawdown",
                    f"{max_drawdown * 100:.2f}%",
                    "The largest peak-to-trough decline in portfolio value.",
                    color="red"
                ) +
                """
                </div>
                """,
                unsafe_allow_html=True
            )

            # Heatmap of Correlations
            st.subheader("Stock Correlation Heatmap")
            correlations = daily_returns.corr()
            heatmap_fig = px.imshow(
                correlations,
                text_auto=True,
                color_continuous_scale="Blues",
                title="Stock Correlation Heatmap",
            )
            st.plotly_chart(heatmap_fig, use_container_width=True)

            # Efficient Frontier and Portfolio Weights
            st.subheader("Efficient Frontier: Portfolio Risk vs Return")
            weights_list = []
            returns = []
            volatilities = []
            for _ in range(5000):
                weights = np.random.random(len(tickers))
                weights /= np.sum(weights)
                weights_list.append(weights)
                port_return, port_volatility = calculate_portfolio_metrics(daily_returns, weights)
                returns.append(port_return)
                volatilities.append(port_volatility)

            returns_percentage = [r * 100 for r in returns]
            volatilities_percentage = [v * 100 for v in volatilities]

            frontier_fig = go.Figure()

            # Simulated portfolios
            frontier_fig.add_trace(go.Scatter(
                x=volatilities_percentage,
                y=returns_percentage,
                mode='markers',
                marker=dict(size=5, color="blue", opacity=0.7),
                name="Simulated Portfolios"
            ))

            # Optimal portfolio
            frontier_fig.add_trace(go.Scatter(
                x=[optimal_volatility_percentage],
                y=[optimal_return_percentage],
                mode='markers',
                marker=dict(color="red", size=12, symbol='star'),
                name="Optimal Portfolio"
            ))

            frontier_fig.update_layout(
                title="Efficient Frontier: Portfolio Risk vs Return",
                xaxis_title="Volatility (Risk) [%]",
                yaxis_title="Expected Return [%]",
                template="plotly_white",
                height=700,
                width=1000,
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
            )
            st.plotly_chart(frontier_fig, use_container_width=True)

            # Add explanation below the Efficient Frontier graph
            st.markdown(f"""
            ### Insights from the Simulation
            After simulating **5000 portfolios**, we have identified an **optimal portfolio** with the following characteristics:
            - **Expected Annual Return**: {optimal_return_percentage:.2f}%
            - **Expected Annual Volatility**: {optimal_volatility_percentage:.2f}%

            By assigning weights as displayed in the allocation table above, you can achieve the best risk-adjusted return for your selected stocks.
            """)

        except Exception as e:
            st.error(f"Error optimizing portfolio: {e}")

# --- Analyze Existing Portfolio ---
if action == "Analyze My Existing Portfolio":
    st.subheader("Analyze Existing Portfolio")
    st.markdown("""
This app allows you to create a stock portfolio, compare it with a benchmark index, 
and analyze important portfolio metrics, including Value at Risk (VaR), Sharpe Ratio, and Sortino Ratio.
""")

    # Sidebar for Portfolio Inputs
    st.sidebar.header("Portfolio Inputs")
    if "portfolio" not in st.session_state:
        st.session_state["portfolio"] = []

    # Stock Input Form
    with st.sidebar.form("portfolio_form"):
        st.subheader("Add Stock to Portfolio")
        stock_ticker = st.text_input("Stock Ticker (e.g., AAPL)", value="AAPL")
        avg_price = st.number_input("Average Price Bought ($)", min_value=0.0, value=100.0)
        quantity = st.number_input("Quantity Bought", min_value=1, value=10)
        add_stock = st.form_submit_button("Add Stock")

        if add_stock:
            st.session_state["portfolio"].append({
                "Ticker": stock_ticker.upper(),
                "Avg Price": avg_price,
                "Quantity": quantity
            })
            st.success(f"Added {stock_ticker.upper()} to your portfolio!")

    # Create `portfolio_df` only if portfolio data exists
    if len(st.session_state["portfolio"]) > 0:
        portfolio_df = pd.DataFrame(st.session_state["portfolio"])
        portfolio_df["Total Investment"] = portfolio_df["Avg Price"] * portfolio_df["Quantity"]
        total_portfolio_value = portfolio_df["Total Investment"].sum()
        portfolio_df["Weight"] = portfolio_df["Total Investment"] / portfolio_df["Total Investment"].sum()

        # Sidebar Inputs
        risk_free_rate_annual = st.sidebar.number_input(
            "Enter Risk-Free Rate (Annual %) :",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.1
        )
        benchmark_options = {
            "S&P 500 (SPY)": "^GSPC",
            "Dow Jones (DIA)": "^DJI",
            "FTSE 100 (FTSE)": "^FTSE",
            "Nikkei 225 (NIKKEI)": "^N225",
            "Euro Stoxx 50 (STOXX50E)": "^STOXX50E",
            "India Nifty 50 (NSEI)": "^NSEI"
        }
        benchmark_index = st.sidebar.selectbox("Choose Benchmark Index", options=list(benchmark_options.keys()))
        benchmark_ticker = benchmark_options[benchmark_index]
        lookback_period = st.sidebar.selectbox("Select Lookback Period", options=["1mo", "3mo", "6mo", "1y", "2y", "5y"])
        var_level = st.sidebar.radio("Select VaR Level (%)", options=[90, 95, 99], index=1)

        if st.sidebar.button("Run Simulation"):
            # Portfolio analysis logic here using `portfolio_df`
            tickers = portfolio_df["Ticker"].tolist()
            try:
                historical_data = yf.download(tickers, period=f"{lookback_period}")["Adj Close"]
            except Exception as e:
                st.error(f"Error fetching data: {e}")
                st.stop()

            # Calculate daily returns
            daily_returns = historical_data.pct_change().dropna()

            # Portfolio metrics
            weights = portfolio_df["Weight"].values
            portfolio_returns = daily_returns.dot(weights)
            portfolio_std = portfolio_returns.std()
            portfolio_mean = portfolio_returns.mean()

            # Fetch Benchmark Data
            try:
                benchmark_data = yf.download(benchmark_ticker, period=f"{lookback_period}")["Adj Close"]

                # Ensure benchmark_data is a single column
                if isinstance(benchmark_data, pd.DataFrame):
                    benchmark_data = benchmark_data.squeeze()  # Convert to Series if necessary

                # Calculate benchmark returns
                benchmark_returns = benchmark_data.pct_change().dropna()

                # Calculate mean and standard deviation
                benchmark_mean = float(benchmark_returns.mean())  # Convert to scalar
                benchmark_std = float(benchmark_returns.std())  # Convert to scalar

                lookback_days_mapping = {
                    "1mo": 21,  # Approx 21 trading days in 1 month
                    "3mo": 63,  # Approx 63 trading days in 3 months
                    "6mo": 126,  # Approx 126 trading days in 6 months
                    "1y": 252,  # Approx 252 trading days in 1 year
                    "2y": 504,  # Approx 504 trading days in 2 years
                    "5y": 1260  # Approx 1260 trading days in 5 years
                }
                lookback_days = lookback_days_mapping[lookback_period]
                risk_free_rate_daily = (risk_free_rate_annual / 100) / 252  # Daily risk-free rate (annualized to daily)
                risk_free_rate_periodic = risk_free_rate_daily * lookback_days  # Adjusted risk-free rate for the period
                # Sharpe Ratio
                portfolio_sharpe = (portfolio_mean - risk_free_rate_periodic) / portfolio_std
                benchmark_sharpe = (benchmark_mean - risk_free_rate_periodic) / benchmark_std

                # Sortino Ratio
                # Portfolio Sortino Ratio
                downside_deviation = portfolio_returns[portfolio_returns < risk_free_rate_periodic].std()
                portfolio_sortino = (portfolio_mean - risk_free_rate_periodic) / downside_deviation
                # Benchmark Sortino Ratio
                benchmark_downside_deviation = benchmark_returns[benchmark_returns < risk_free_rate_periodic].std()
                benchmark_sortino = (benchmark_mean - risk_free_rate_periodic) / benchmark_downside_deviation

                # Value at Risk (VaR)
                z_score = {90: 1.28, 95: 1.645, 99: 2.33}[var_level]
                portfolio_var = z_score * portfolio_std * total_portfolio_value
                benchmark_var = z_score * benchmark_returns.std() * total_portfolio_value


                # Maximum Drawdown Calculation
                def calculate_max_drawdown(cumulative_returns):
                    """Calculate Maximum Drawdown."""
                    drawdowns = cumulative_returns / cumulative_returns.cummax() - 1
                    return drawdowns.min()


                # Calculate cumulative returns
                cumulative_portfolio = (1 + portfolio_returns).cumprod()
                cumulative_benchmark = (1 + benchmark_returns).cumprod()

                # Calculate Maximum Drawdown for Portfolio and Benchmark
                portfolio_mdd = calculate_max_drawdown(cumulative_portfolio)
                benchmark_mdd = calculate_max_drawdown(cumulative_benchmark)
                # Portfolio Alpha Calculation
                portfolio_alpha = portfolio_mean - (
                        risk_free_rate_periodic + portfolio_std * (benchmark_mean - risk_free_rate_periodic))

                # Display the Portfolio
                # Display the Portfolio
                portfolio_df = pd.DataFrame(st.session_state["portfolio"])
                if not portfolio_df.empty:
                    # Calculate Total Investment
                    portfolio_df["Total Investment"] = portfolio_df["Avg Price"] * portfolio_df["Quantity"]
                    total_portfolio_value = portfolio_df["Total Investment"].sum()
                    portfolio_df["Weight"] = portfolio_df["Total Investment"] / total_portfolio_value

                    # Fetch Current Market Prices
                    try:
                        tickers = portfolio_df["Ticker"].tolist()
                        current_prices = yf.download(tickers, period="1d")["Adj Close"].iloc[
                            -1]  # Get the latest adjusted close
                        portfolio_df["Current Price"] = portfolio_df["Ticker"].map(current_prices)
                        portfolio_df["Market Value"] = portfolio_df["Current Price"] * portfolio_df["Quantity"]
                        portfolio_df["Unrealized PnL (%)"] = ((portfolio_df["Market Value"] - portfolio_df[
                            "Total Investment"]) / portfolio_df["Total Investment"]) * 100
                        total_market_value = portfolio_df["Market Value"].sum()
                        portfolio_percentage_return = ((
                                                                   total_market_value - total_portfolio_value) / total_portfolio_value) * 100
                    except Exception as e:
                        st.error(f"Error fetching current prices: {e}")
                        st.stop()

                    # Display Portfolio Details
                    st.subheader("Portfolio Details")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Portfolio Value ($)", f"{total_portfolio_value:,.2f}")
                    with col2:
                        st.metric("Current Market Value ($)", f"{total_market_value:,.2f}")
                    with col3:
                        st.metric("Portfolio Return (%)", f"{portfolio_percentage_return:.2f}%",
                                  "Unrealized return on portfolio")

                    # Display Dataframe
                    st.dataframe(portfolio_df)

                    # Add Portfolio Pie Chart
                    portfolio_pie = px.pie(
                        portfolio_df,
                        names="Ticker",
                        values="Market Value",
                        title="Portfolio Allocation by Market Value",
                        hole=0.4
                    )
                    st.plotly_chart(portfolio_pie, use_container_width=True)
                else:
                    st.info("Your portfolio is empty. Add stocks to begin.")

                # option 4
                # Display Metrics with Enhanced Layout
                st.subheader("Metrics")

                # Create Tabs for Portfolio and Benchmark Metrics
                tab1, tab2 = st.tabs(["📈 Portfolio Metrics", "📊 Benchmark Metrics"])


                # Define common styles for black text and value colors
                def generate_card(title, value, description, color="black", bg_color="#f9f9f9"):
                    return f"""
                                <div style="
                                    background-color: {bg_color};
                                    color: black;
                                    padding: 15px;
                                    border-radius: 10px;
                                    margin-bottom: 10px;
                                    font-family: Arial, sans-serif;
                                ">
                                    <h3 style="margin: 0; color: black;">{title}</h3>
                                    <p style="font-size: 20px; font-weight: bold; color: {color};">{value}</p>
                                    <small style="color: gray;">{description}</small>
                                </div>
                                """


                # Portfolio Metrics Tab
                with tab1:
                    st.markdown("### Portfolio Metrics")
                    col1, col2 = st.columns(2)
                    with col1:
                        # Average Daily Return
                        avg_return_color = "green" if portfolio_mean > 0 else "red"
                        st.markdown(
                            generate_card(
                                "📈 Average Daily Return",
                                f"{portfolio_mean * 100:.2f}%",
                                "Expected daily return of the portfolio.",
                                color=avg_return_color
                            ),
                            unsafe_allow_html=True
                        )

                        # Portfolio Volatility
                        st.markdown(
                            generate_card(
                                "Volatility (Std Dev)",
                                f"{portfolio_std * 100:.2f}%",
                                "Indicates the riskiness of the portfolio."
                            ),
                            unsafe_allow_html=True
                        )

                        # Maximum Drawdown
                        st.markdown(
                            generate_card(
                                "Maximum Drawdown",
                                f"{portfolio_mdd * 100:.2f}%",
                                "Largest peak-to-trough portfolio loss."
                            ),
                            unsafe_allow_html=True
                        )

                    with col2:
                        # Sharpe Ratio
                        sharpe_color = "green" if portfolio_sharpe > 1 else "red"
                        st.markdown(
                            generate_card(
                                "📊 Sharpe Ratio",
                                f"{portfolio_sharpe:.2f}",
                                "Higher Sharpe Ratio indicates better risk-adjusted returns.",
                                color=sharpe_color
                            ),
                            unsafe_allow_html=True
                        )

                        # Sortino Ratio
                        st.markdown(
                            generate_card(
                                "Sortino Ratio",
                                f"{portfolio_sortino:.2f}",
                                "Focuses on downside risk; higher is better."
                            ),
                            unsafe_allow_html=True
                        )

                        # Alpha
                        alpha_color = "green" if portfolio_alpha > 0 else "red"
                        st.markdown(
                            generate_card(
                                "🧮 Alpha",
                                f"{portfolio_alpha * 100:.2f}%",
                                "Alpha measures the portfolio's risk-adjusted performance relative to the benchmark. "
                                "A positive alpha indicates outperformance, while a negative alpha indicates underperformance.",
                                color=alpha_color
                            ),
                            unsafe_allow_html=True
                        )

                # Benchmark Metrics Tab
                with tab2:
                    st.markdown("### Benchmark Metrics")
                    col1, col2 = st.columns(2)
                    with col1:
                        # Average Daily Return
                        avg_return_color = "green" if benchmark_mean > 0 else "red"
                        st.markdown(
                            generate_card(
                                "📈 Average Daily Return",
                                f"{benchmark_mean * 100:.2f}%",
                                "Expected daily return of the benchmark.",
                                color=avg_return_color
                            ),
                            unsafe_allow_html=True
                        )

                        # Benchmark Volatility
                        st.markdown(
                            generate_card(
                                "Benchmark Volatility (Std Dev)",
                                f"{benchmark_std * 100:.2f}%",
                                "Indicates the riskiness of the benchmark."
                            ),
                            unsafe_allow_html=True
                        )

                        # Maximum Drawdown
                        st.markdown(
                            generate_card(
                                "Maximum Drawdown",
                                f"{benchmark_mdd * 100:.2f}%",
                                "Largest peak-to-trough benchmark loss."
                            ),
                            unsafe_allow_html=True
                        )

                    with col2:
                        # Sharpe Ratio
                        sharpe_color = "green" if benchmark_sharpe > 1 else "red"
                        st.markdown(
                            generate_card(
                                "📊 Sharpe Ratio",
                                f"{benchmark_sharpe:.2f}",
                                "Higher Sharpe Ratio indicates better risk-adjusted returns.",
                                color=sharpe_color
                            ),
                            unsafe_allow_html=True
                        )

                        # Sortino Ratio
                        st.markdown(
                            generate_card(
                                "Sortino Ratio",
                                f"{benchmark_sortino:.2f}",
                                "Focuses on downside risk; higher is better."
                            ),
                            unsafe_allow_html=True
                        )

                # Cumulative Returns Comparison
                st.subheader("Cumulative Returns Comparison")
                cumulative_portfolio = (1 + portfolio_returns).cumprod()
                cumulative_benchmark = (1 + benchmark_returns).cumprod()
                comparison_fig = go.Figure()
                comparison_fig.add_trace(go.Scatter(
                    x=cumulative_portfolio.index, y=cumulative_portfolio.values,
                    mode="lines", name="Portfolio", line=dict(width=3, color="blue")
                ))
                comparison_fig.add_trace(go.Scatter(
                    x=cumulative_benchmark.index, y=cumulative_benchmark.values,
                    mode="lines", name=benchmark_index, line=dict(width=3, color="orange")
                ))
                comparison_fig.update_layout(
                    title="Portfolio vs Benchmark Cumulative Returns",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Returns",
                    template="plotly_white"
                )
                st.plotly_chart(comparison_fig, use_container_width=True)

                # Value at Risk (VaR) Display
                st.subheader("Value at Risk (VaR)")
                st.metric(f"Portfolio VaR ({var_level}% Confidence)", f"${portfolio_var:,.2f}")
                st.metric(f"Benchmark VaR ({var_level}% Confidence)", f"${benchmark_var:,.2f}")
                # Value at Risk (VaR) Histograms
                # st.subheader("Value at Risk (VaR)")

                # Calculate VaR thresholds
                portfolio_var_threshold = -z_score * portfolio_std
                benchmark_var_threshold = -z_score * benchmark_std

                # Create Portfolio VaR Histogram
                portfolio_hist_fig = go.Figure()

                # Add portfolio returns histogram
                portfolio_hist_fig.add_trace(
                    go.Histogram(
                        x=portfolio_returns,
                        nbinsx=50,
                        marker_color="blue",
                        opacity=0.7,
                        name="Portfolio Returns"
                    )
                )

                # Add a vertical line for Portfolio VaR threshold
                portfolio_hist_fig.add_vline(
                    x=portfolio_var_threshold,
                    line_width=3,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Portfolio VaR ({var_level}%) = {portfolio_var_threshold:.2%}",
                    annotation_position="top left"
                )

                portfolio_hist_fig.update_layout(
                    title=f"Portfolio Returns Distribution with VaR ({var_level}% Confidence)",
                    xaxis_title="Daily Returns",
                    yaxis_title="Frequency",
                    template="plotly_white",
                    showlegend=False
                )

                # Create Benchmark VaR Histogram
                benchmark_hist_fig = go.Figure()

                # Add benchmark returns histogram
                benchmark_hist_fig.add_trace(
                    go.Histogram(
                        x=benchmark_returns,
                        nbinsx=50,
                        marker_color="orange",
                        opacity=0.7,
                        name="Benchmark Returns"
                    )
                )

                # Add a vertical line for Benchmark VaR threshold
                benchmark_hist_fig.add_vline(
                    x=benchmark_var_threshold,
                    line_width=3,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Benchmark VaR ({var_level}%) = {benchmark_var_threshold:.2%}",
                    annotation_position="top left"
                )

                benchmark_hist_fig.update_layout(
                    title=f"Benchmark Returns Distribution with VaR ({var_level}% Confidence)",
                    xaxis_title="Daily Returns",
                    yaxis_title="Frequency",
                    template="plotly_white",
                    showlegend=False
                )

                # Display Portfolio and Benchmark VaR Histograms Side-by-Side
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Portfolio VaR Histogram")
                    st.plotly_chart(portfolio_hist_fig, use_container_width=True)

                with col2:
                    st.markdown("### Benchmark VaR Histogram")
                    st.plotly_chart(benchmark_hist_fig, use_container_width=True)

                st.markdown("""
                            ### **Interpretation of Value at Risk (VaR)**


                            - A **95% VaR** of **$10,000** means:
                              - There is a **95% probability** that the portfolio will **not lose more than $10,000** in a single day.
                              - Conversely, there is a **5% probability** that losses **could exceed $10,000**.


                            #### **Limitations**:
                            - **Not Absolute**: VaR does not predict the magnitude of losses beyond the specified threshold (e.g., what happens in the worst 5% of cases).
                            - VaRAssumes Normal Market Conditions, It may not capture extreme market events like financial crises.


                            ---
                            **Note**: VaR should be used alongside other risk metrics (like Maximum Drawdown) for a comprehensive risk analysis.
                            """)


            except Exception as e:
                st.error(f"Error in simulation: {e}")



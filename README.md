# 📊 Portfolio Management Application

An interactive Streamlit-based application designed to help users create, optimize, and analyze investment portfolios. The application provides advanced financial metrics, portfolio optimization, and risk analysis tools to assist in informed decision-making for both beginners and experienced investors.

---

## 🚀 Features

- **Portfolio Optimization**:
  - Generate an optimal portfolio allocation based on historical data.
  - Calculate portfolio metrics, including Sharpe Ratio, Volatility, and Expected Returns.
  - Visualize the Efficient Frontier for portfolio risk vs. return.

- **Existing Portfolio Analysis**:
  - Analyze and monitor your existing stock portfolio.
  - Compare portfolio performance against benchmark indices like S&P 500, Dow Jones, and Nifty 50.
  - Calculate key metrics like VaR (Value at Risk), Maximum Drawdown, and Sortino Ratio.

- **Interactive Visualizations**:
  - Portfolio allocation pie charts.
  - Stock correlation heatmaps.
  - Cumulative returns comparisons with benchmarks.
  - Efficient Frontier simulation with 5000 portfolios.

- **Risk Analysis**:
  - Compute Value at Risk (VaR) with customizable confidence levels.
  - Evaluate portfolio drawdowns and performance metrics.

---

## 🛠️ Technologies Used

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Data Fetching**: [Yahoo Finance API](https://pypi.org/project/yfinance/)
- **Visualization**: [Plotly](https://plotly.com/python/) (Express and Graph Objects)
- **Optimization**: [Scipy](https://scipy.org/) for portfolio weight calculation
- **Backend Logic**: Python with Numpy and Pandas for data processing

---

## 🖥️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/portfolio-management-app.git
   cd portfolio-management-app
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

4. Access the application in your browser at `http://localhost:8501`.

---

## 📄 Usage

### 1. Create an Optimal Portfolio
- Input the stock tickers, investment amount, and desired lookback period.
- Click on **Run Optimization** to generate an optimal allocation.
- Review metrics like Sharpe Ratio and Maximum Drawdown.
- Visualize allocation, Efficient Frontier, and correlations.

### 2. Analyze Existing Portfolio
- Add stock details (ticker, average price, quantity).
- Select a benchmark index and desired lookback period.
- Analyze portfolio vs. benchmark performance and calculate key metrics.
- View cumulative returns, correlation heatmaps, and VaR histograms.

---

## 🎯 Key Metrics Calculated
- **Sharpe Ratio**: Measures risk-adjusted returns.
- **Sortino Ratio**: Focuses on downside risk.
- **Value at Risk (VaR)**: Estimates potential portfolio loss at a given confidence level.
- **Maximum Drawdown (MDD)**: Largest portfolio decline from a peak.

---

## 🏆 Why Use This Application?
- Streamline your investment decisions with actionable insights.
- Optimize portfolios for maximum returns under given risk constraints.
- Gain an edge in risk management and portfolio evaluation.

---

## 🧩 Contribution
Contributions are welcome! If you'd like to enhance the application, feel free to fork the repo and submit a pull request.

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a Pull Request.

---

## 🔒 License
This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Acknowledgements
- **Yahoo Finance** for data API.
- **Streamlit** for providing an excellent framework for interactive web apps.
- Financial community for inspiration and feedback.

---

## 📬 Contact
For any queries or feedback, please reach out:
- **Email**: [your.email@example.com](mailto:your.email@example.com)
- **GitHub**: [your-username](https://github.com/your-username) 

---

Feel free to explore and enhance the application for your financial analysis needs! 🌟

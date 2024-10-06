import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime, timedelta

# Initialize exchange
exchange = ccxt.kucoin()

# Load all markets
markets = exchange.load_markets()

# Extract symbols of interest (e.g., trading pairs with USDT)
symbols = [symbol for symbol in markets if '/USDT' in symbol]  # Modify as needed

# Define the exponential growth function
def exp_growth(t, P0, r):
    return P0 * np.exp(r * t)

# Define the exponential decay function
def exp_decay(t, P0, lam):
    return P0 * np.exp(-lam * t)

# Calculate the date for 500 days ago
since_date = datetime.now() - timedelta(days=500)
since = exchange.parse8601(since_date.isoformat() + 'Z')  # Use UTC format

# Initialize an empty list to collect data for the CSV
csv_data = []

# Loop through each symbol
for symbol in symbols:
    try:
        # Fetch OHLCV data for the past 500 days
        ohlcv = exchange.fetch_ohlcv(symbol, '1d', since)

        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Prepare time variable (normalized)
        df['time'] = (df.index - df.index.min()).days

        # Identify the highest price and its index
        highest_price = df['close'].max()
        highest_price_idx = df['close'].idxmax()
        time_at_high = df['time'][highest_price_idx]

        # Fit exponential growth model up to the highest price
        initial_guess_growth = [df['close'].iloc[0], 0.01]  # Initial price and growth rate
        growth_segment = df[df['time'] <= time_at_high]  # Data up to the highest price
        params_growth, _ = curve_fit(exp_growth, growth_segment['time'], growth_segment['close'], p0=initial_guess_growth)
        fitted_growth = exp_growth(growth_segment['time'], *params_growth)

        # Calculate R-squared for growth fit
        residuals_growth = growth_segment['close'] - fitted_growth
        ss_res_growth = np.sum(residuals_growth**2)
        ss_tot_growth = np.sum((growth_segment['close'] - np.mean(growth_segment['close']))**2)
        r_squared_growth = 1 - (ss_res_growth / ss_tot_growth)

        # Fit exponential decay model from the highest price onward
        decay_segment = df[df['time'] > time_at_high]  # Data after the highest price
        if not decay_segment.empty:
            initial_guess_decay = [highest_price, 0.01]  # Start from highest price
            params_decay, _ = curve_fit(exp_decay, decay_segment['time'], decay_segment['close'], p0=initial_guess_decay)
            fitted_decay = exp_decay(decay_segment['time'], *params_decay)

            # Calculate R-squared for decay fit
            residuals_decay = decay_segment['close'] - fitted_decay
            ss_res_decay = np.sum(residuals_decay**2)
            ss_tot_decay = np.sum((decay_segment['close'] - np.mean(decay_segment['close']))**2)
            r_squared_decay = 1 - (ss_res_decay / ss_tot_decay)

            # Calculate score (growth R² - decay R²)
            score = r_squared_growth - r_squared_decay

            # Create trading link
            link = f'https://www.kucoin.com/trade/{symbol}-USDT'
        else:
            r_squared_decay = None  # No decay model fitted
            score = None
            link = f'https://www.kucoin.com/trade/{symbol}-USDT'

        # Append data to the CSV data list
        csv_data.append({
            'symbol': symbol,
            'exchange': 'KuCoin',  # Assuming KuCoin since it's part of the link
            'growth_r_squared': r_squared_growth if r_squared_growth is not None else 'N/A',
            'decay_r_squared': r_squared_decay if r_squared_decay is not None else 'N/A',
            'score': score if score is not None else 'N/A',
            'link': link
        })

        # Plot original price data and fitted models
        plt.figure(figsize=(12, 6))
        plt.plot(df['time'], df['close'], label='Original Price', color='blue')

        # Plot the exponential growth line
        plt.plot(growth_segment['time'], fitted_growth, label='Exponential Growth Fit', color='green')

        # Plot the exponential decay line
        if not decay_segment.empty:
            plt.plot(decay_segment['time'], fitted_decay, label='Exponential Decay Fit', color='red')

        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Optional: horizontal line at y=0
        plt.title(f'{symbol} Price with Exponential Growth and Decay')
        plt.xlabel('Days since first observation')
        plt.ylabel('Price (USDT)')
        plt.legend()
        plt.grid()

        # Annotate R-squared values on the chart
        plt.text(0.05, 0.95, f'Growth R²: {r_squared_growth:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', color='green')
        if not decay_segment.empty:
            plt.text(0.05, 0.90, f'Decay R²: {r_squared_decay:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', color='red')

        plt.show()

        # Print parameters for the fits and R-squared values
        print(f"{symbol} Growth Fit - Initial Price: {params_growth[0]:.2f}, Growth Rate: {params_growth[1]:.4f}, R-squared: {r_squared_growth:.4f}")
        if not decay_segment.empty:
            print(f"{symbol} Decay Fit - Initial Price: {params_decay[0]:.2f}, Decay Rate: {params_decay[1]:.4f}, R-squared: {r_squared_decay:.4f}")
        else:
            print(f"{symbol} No decay fit available.")

    except Exception as e:
        print(f"Error processing {symbol}: {e}")

# Create DataFrame from the collected data
csv_df = pd.DataFrame(csv_data)

# Save the DataFrame to a CSV file
csv_filename = 'crypto_analysis_report.csv'
csv_df.to_csv(csv_filename, index=False)
print(f"CSV file '{csv_filename}' created successfully.")

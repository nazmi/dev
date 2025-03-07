import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

def is_cryptocurrency(ticker):
    """
    Determine if the ticker is a cryptocurrency by checking if it contains a hyphen
    Most crypto tickers on Yahoo Finance are in the format BTC-USD, ETH-USD, etc.
    """
    return '-' in ticker

def get_price_data(ticker, start_date, end_date):
    """
    Fetch historical price data for the given ticker
    Use appropriate settings based on asset type (stock or cryptocurrency)
    """
    is_crypto = is_cryptocurrency(ticker)
    asset_type = "cryptocurrency" if is_crypto else "stock"
    
    print(f"Downloading {asset_type} data for {ticker} from {start_date} to {end_date}...")
    
    # For stocks, use auto_adjust=False to get both adjusted and unadjusted prices
    # For crypto, use auto_adjust=True as there are no splits to worry about
    price_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=is_crypto)
    
    print(f"Retrieved {len(price_data)} data points for {ticker}")
    return price_data, is_crypto

def get_scalar_value(df, idx, column):
    """
    Safely extract a scalar value from a DataFrame
    This handles different pandas versions and behaviors
    """
    try:
        # Try direct .at access first (most efficient for scalar)
        return float(df.at[idx, column])
    except Exception:
        try:
            # Try .loc access with .iloc[0] as fallback
            return float(df.loc[idx, column].iloc[0])
        except Exception:
            try:
                # Try .loc with values[0] as second fallback
                return float(df.loc[idx, column].values[0])
            except Exception as e:
                # If all methods fail, print diagnostic info and raise
                print(f"Error extracting {column} at {idx}: {e}")
                print(f"Value type: {type(df.loc[idx, column])}")
                if hasattr(df.loc[idx, column], 'shape'):
                    print(f"Shape: {df.loc[idx, column].shape}")
                raise

def get_financial_report_date(ticker, year):
    """
    Get the approximate financial report date for a company
    
    Parameters:
    ticker (str): Stock ticker symbol
    year (int): Year to get report date for
    
    Returns:
    datetime: Approximate financial report date
    """
    # For AAPL - generally late January
    if ticker == 'AAPL':
        # Apple typically reports Q4 in late January
        return datetime(year, 1, 28)
    # For ETFs like SOXX and VTI - typically early February
    elif ticker in ['SOXX', 'VTI']:
        return datetime(year, 2, 5)
    # Default for other companies - assume early February
    else:
        return datetime(year, 2, 1)

def get_trading_day_before_date(price_data, target_date):
    """
    Find the closest trading day before a given date
    
    Parameters:
    price_data (DataFrame): Historical price data
    target_date (datetime): Target date
    
    Returns:
    pandas.Timestamp: Closest trading day before target_date
    """
    # Convert target_date to string format YYYY-MM-DD for comparison
    target_date_str = target_date.strftime('%Y-%m-%d')
    
    # Filter to dates before target_date
    before_dates = price_data[price_data.index < target_date_str]
    
    if len(before_dates) == 0:
        # If no dates before target, get earliest available date
        return price_data.index[0]
    
    # Get the last trading day before target date
    return before_dates.index[-1]

def simulate_timing_strategies(ticker, years=5, annual_investment=2000, daily_investment=20, invest_on_financial_day=False):
    """
    Simulate different market timing strategies for stocks or cryptocurrencies:
    1. Investing at yearly low
    2. Investing on January 1st or day before financial report (for stocks)
    3. Investing at yearly high
    4. Investing daily (dollar-cost averaging)
    
    Parameters:
    ticker (str): Ticker symbol (stock or cryptocurrency)
    years (int): Number of years to analyze
    annual_investment (float): Amount to invest each year for strategies 1-3
    daily_investment (float): Amount to invest each trading day for strategy 4
    invest_on_financial_day (bool): If True, invest day before financial report instead of January 1st
                                   (only applies to stocks, not cryptocurrencies)
    
    Returns:
    dict: Final values for each strategy
    dict: Information about investments including total amounts
    """
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365.25 * years)
    
    # Format dates as strings for yfinance
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # Get price data and determine if it's a cryptocurrency
    price_data, is_crypto = get_price_data(ticker, start_date_str, end_date_str)
    
    if len(price_data) == 0:
        print(f"Warning: No data found for {ticker}")
        return {
            'Lowest Price': 0,
            'Jan1/FinDay': 0, 
            'Highest Price': 0,
            'Daily $20': 0
        }, {}
    
    # Initialize results and tracking for units (shares or coins)
    units = {
        'Lowest Price': 0,
        'Jan1/FinDay': 0,
        'Highest Price': 0,
        'Daily $20': 0
    }
    
    # Calculate yearly performance for each strategy
    current_year = start_date.year
    end_year = min(end_date.year, current_year + years - 1)
    
    # Set description based on asset type
    asset_type = "cryptocurrency" if is_crypto else "stock"
    unit_type = "coins" if is_crypto else "shares"
    
    # For crypto, we can't use financial report dates, so force January 1st
    if is_crypto:
        invest_on_financial_day = False
    
    jan_fin_label = "Financial Report Day" if invest_on_financial_day else "January 1st"
    print(f"Analyzing years from {current_year} to {end_year} for {ticker} ({asset_type})")
    print(f"Strategy 2 is investing on {jan_fin_label}")
    
    # First process yearly investment strategies (1-3)
    for year in range(current_year, end_year + 1):
        try:
            # Get data for current year
            year_data = price_data[price_data.index.year == year]
            
            if len(year_data) == 0:
                print(f"  No data for {ticker} in {year}, skipping")
                continue
                
            # Get lowest price of the year
            lowest_idx = year_data['Low'].idxmin()
            lowest_price = get_scalar_value(year_data, lowest_idx, 'Low')
            
            # Get highest price of the year
            highest_idx = year_data['High'].idxmax()
            highest_price = get_scalar_value(year_data, highest_idx, 'High')
            
            # Get January 1st or Financial Report Day price
            if invest_on_financial_day and not is_crypto:
                # Get the financial report date for this year
                fin_report_date = get_financial_report_date(ticker, year)
                
                # Find the trading day before the financial report
                fin_day_idx = get_trading_day_before_date(price_data, fin_report_date)
                fin_day_price = get_scalar_value(price_data, fin_day_idx, 'Open')
                
                jan_fin_idx = fin_day_idx
                jan_fin_price = fin_day_price
                date_description = f"Financial Day ({fin_day_idx.strftime('%Y-%m-%d')})"
            else:
                # Use January 1st (or closest trading day)
                jan_dates = price_data[price_data.index.year == year].index
                jan_dates = [d for d in jan_dates if d.month == 1]
                
                if jan_dates:
                    jan_idx = jan_dates[0]
                    jan_price = get_scalar_value(price_data, jan_idx, 'Open')
                else:
                    # If no January data, use first available date
                    jan_idx = year_data.index[0]
                    jan_price = get_scalar_value(year_data, jan_idx, 'Open')
                
                jan_fin_idx = jan_idx
                jan_fin_price = jan_price
                date_description = "January 1st"
            
            # Calculate units (shares or coins) purchased
            lowest_units = annual_investment / lowest_price
            jan_fin_units = annual_investment / jan_fin_price
            highest_units = annual_investment / highest_price
            
            # For stocks, apply split adjustments
            if not is_crypto:
                # Store purchase information for later adjustment
                purchase_info = {
                    'Lowest Price': {
                        'date': lowest_idx,
                        'price': lowest_price,
                        'units': lowest_units
                    },
                    'Jan1/FinDay': {
                        'date': jan_fin_idx,
                        'price': jan_fin_price,
                        'units': jan_fin_units
                    },
                    'Highest Price': {
                        'date': highest_idx,
                        'price': highest_price,
                        'units': highest_units
                    }
                }
                
                # Record purchases with adjustment for splits
                for strategy, info in purchase_info.items():
                    # Get the split adjustment factor by comparing Adj Close to Close on purchase date
                    purchase_date = info['date']
                    
                    if 'Adj Close' in price_data.columns and 'Close' in price_data.columns:
                        adj_close = get_scalar_value(price_data, purchase_date, 'Adj Close')
                        close = get_scalar_value(price_data, purchase_date, 'Close')
                        
                        if close > 0:  # Avoid division by zero
                            split_factor = adj_close / close
                        else:
                            split_factor = 1.0
                    else:
                        split_factor = 1.0
                    
                    # Apply the split adjustment to share count
                    split_adjusted_units = info['units'] * split_factor
                    units[strategy] += split_adjusted_units
                    
                    # Use proper strategy name for printing
                    strategy_name = date_description if strategy == 'Jan1/FinDay' else strategy
                    
                    print(f"  {year} {strategy_name}: Price=${info['price']:.2f}, {unit_type}={info['units']:.4f}, " +
                         f"Split-Adjusted {unit_type}={split_adjusted_units:.4f}")
            else:
                # For crypto, no split adjustments needed
                units['Lowest Price'] += lowest_units
                units['Jan1/FinDay'] += jan_fin_units
                units['Highest Price'] += highest_units
                
                # Print crypto details with more decimal places
                price_format = ".2f" if lowest_price >= 1.0 else ".8f"
                print(f"  {year} Lowest Price: ${lowest_price:{price_format}}, {unit_type}={lowest_units:.8f}")
                print(f"  {year} {date_description}: ${jan_fin_price:{price_format}}, {unit_type}={jan_fin_units:.8f}")
                print(f"  {year} Highest Price: ${highest_price:{price_format}}, {unit_type}={highest_units:.8f}")
                
        except Exception as e:
            print(f"  Error processing {year} data for {ticker}: {e}")
            continue
    
    # Process daily investment strategy (4)
    daily_total_invested = 0
    total_trading_days = 0
    
    try:
        print(f"\nProcessing daily investments of ${daily_investment} for {ticker}...")
        daily_unit_count = 0
        
        # Iterate through each trading day
        for date, row in price_data.iterrows():
            try:
                # Get price and calculate units purchased
                daily_price = get_scalar_value(price_data, date, 'Open')
                daily_units = daily_investment / daily_price
                
                if not is_crypto:
                    # For stocks, apply split adjustments
                    if 'Adj Close' in price_data.columns and 'Close' in price_data.columns:
                        adj_close = get_scalar_value(price_data, date, 'Adj Close')
                        close = get_scalar_value(price_data, date, 'Close')
                        
                        if close > 0:
                            split_factor = adj_close / close
                        else:
                            split_factor = 1.0
                        
                        # Add split-adjusted units
                        split_adjusted_daily_units = daily_units * split_factor
                        daily_unit_count += split_adjusted_daily_units
                    else:
                        daily_unit_count += daily_units
                else:
                    # For crypto, no split adjustments needed
                    daily_unit_count += daily_units
                
                daily_total_invested += daily_investment
                total_trading_days += 1
                
                # Print progress periodically (first day of each year)
                if date.day == date.month == 1:
                    print(f"  Daily investing progress to {date.date()}: {daily_unit_count:.8f} {unit_type}, ${daily_total_invested:.2f} invested")
                
            except Exception as e:
                # Skip any problematic days
                continue
        
        units['Daily $20'] = daily_unit_count
        print(f"  Completed daily investing: {daily_unit_count:.8f} {unit_type}, ${daily_total_invested:.2f} invested")
        
        # Calculate expected trading days based on asset type
        expected_days = years * (365 if is_crypto else 252)
        print(f"  Total trading days: {total_trading_days} (vs. estimated {expected_days} days)")
    
    except Exception as e:
        print(f"Error processing daily investments for {ticker}: {e}")
    
    try:
        # Calculate final values
        if 'Adj Close' in price_data.columns:
            latest_price = get_scalar_value(price_data.iloc[[-1]], price_data.index[-1], 'Adj Close')
            price_type = "adjusted close"
        else:
            latest_price = get_scalar_value(price_data.iloc[[-1]], price_data.index[-1], 'Close')
            price_type = "close"
            
        print(f"Latest {price_type} price for {ticker}: ${latest_price:.2f}")
        
        results = {}
        for strategy, total_units in units.items():
            final_value = total_units * latest_price
            results[strategy] = final_value
            
            # Use proper strategy name for printing
            strategy_name = jan_fin_label if strategy == 'Jan1/FinDay' else strategy
            print(f"  {strategy_name}: {total_units:.8f} {unit_type}, value=${final_value:.2f}")
        
        # Create investment info dictionary with total invested for each strategy
        investment_info = {
            'Lowest Price': annual_investment * (end_year - current_year + 1),
            'Jan1/FinDay': annual_investment * (end_year - current_year + 1),
            'Highest Price': annual_investment * (end_year - current_year + 1),
            'Daily $20': daily_total_invested,
            'trading_days': total_trading_days,
            'invest_on_financial_day': invest_on_financial_day,
            'is_crypto': is_crypto
        }
        
        return results, investment_info
    
    except Exception as e:
        print(f"Error calculating final values for {ticker}: {e}")
        # Return zeros as fallback
        return {
            'Lowest Price': 0,
            'Jan1/FinDay': 0, 
            'Highest Price': 0,
            'Daily $20': 0
        }, {}

def analyze_assets(tickers, years=5, annual_investment=2000, daily_investment=20, invest_on_financial_day=False):
    """
    Analyze multiple assets (stocks or cryptocurrencies) and compare their performance across different strategies
    
    Parameters:
    tickers (list): List of ticker symbols
    years (int): Number of years to analyze
    annual_investment (float): Amount to invest each year for strategies 1-3
    daily_investment (float): Amount to invest each trading day for strategy 4
    invest_on_financial_day (bool): If True, invest day before financial report instead of January 1st
    
    Returns:
    pandas.DataFrame: Results for all tickers and strategies
    """
    all_results = {}
    all_investment_info = {}
    
    for ticker in tickers:
        try:
            print(f"\nAnalyzing {ticker}...")
            results, investment_info = simulate_timing_strategies(
                ticker, years, annual_investment, daily_investment, invest_on_financial_day
            )
            all_results[ticker] = results
            all_investment_info[ticker] = investment_info
            
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            all_results[ticker] = {
                'Lowest Price': 0,
                'Jan1/FinDay': 0, 
                'Highest Price': 0,
                'Daily $20': 0
            }
            all_investment_info[ticker] = {}
    
    # Convert to DataFrame for easier visualization
    results_list = []
    
    for ticker in all_results:
        ticker_results = all_results[ticker]
        ticker_info = all_investment_info.get(ticker, {})
        is_crypto = ticker_info.get('is_crypto', False)
        
        for strategy, value in ticker_results.items():
            # Calculate total invested amount based on strategy
            if strategy == 'Daily $20':
                # Use actual total invested from our simulation
                total_invested_float = float(ticker_info.get('Daily $20', 0))
            else:
                # For yearly strategies
                total_invested_float = float(ticker_info.get(strategy, annual_investment * years))
            
            # Ensure values are proper floats
            value_float = float(value) if value is not None else 0.0
            
            # Calculate return percentage
            if total_invested_float > 0:
                return_pct = (value_float - total_invested_float) / total_invested_float * 100
            else:
                return_pct = 0.0
            
            # Get the proper strategy label
            if strategy == 'Jan1/FinDay':
                if ticker_info.get('invest_on_financial_day', False) and not is_crypto:
                    strategy_label = 'Financial Report Day'
                else:
                    strategy_label = 'January 1st'
            else:
                strategy_label = strategy
            
            # Set asset type
            asset_type = "Crypto" if is_crypto else "Stock"
            
            results_list.append({
                'Ticker': ticker,
                'Type': asset_type,
                'Strategy': strategy_label,
                'Final Value': value_float,
                'Total Invested': total_invested_float,
                'Return': return_pct
            })
    
    # Create DataFrame
    if results_list:
        results_df = pd.DataFrame(results_list)
        return results_df
    else:
        # Return empty DataFrame with correct columns if no results
        return pd.DataFrame(columns=['Ticker', 'Type', 'Strategy', 'Final Value', 'Total Invested', 'Return'])

def plot_results(results_df):
    """
    Create a bar plot of returns, sorted from highest to lowest
    
    Parameters:
    results_df (pandas.DataFrame): Results dataframe
    """
    if results_df.empty:
        print("No results to plot")
        return
    
    if 'Ticker' not in results_df.columns or 'Strategy' not in results_df.columns or 'Return' not in results_df.columns:
        print("Required columns missing from results DataFrame")
        print(f"Available columns: {results_df.columns.tolist()}")
        return
        
    # Create a new column for the x-axis labels
    results_df['Label'] = results_df['Ticker'] + ' - ' + results_df['Strategy']
    
    print("\nReturn values for sorting:")
    for index, row in results_df.iterrows():
        print(f"{row['Label']} ({row['Type']}): {row['Return']:.2f}%")
    
    # Sort by return (highest to lowest)
    sorted_df = results_df.sort_values('Return', ascending=False)
    
    # Create plot
    plt.figure(figsize=(16, 10))
    bars = plt.bar(sorted_df['Label'], sorted_df['Return'])
    
    # Add colors by strategy
    colors = {
        'Lowest Price': 'green', 
        'January 1st': 'blue',
        'Financial Report Day': 'cyan',
        'Highest Price': 'red',
        'Daily $20': 'purple'
    }
    
    # Add patterns by asset type
    hatches = {
        'Stock': '',
        'Crypto': '/'
    }
    
    for i, bar in enumerate(bars):
        strategy = sorted_df.iloc[i]['Strategy']
        asset_type = sorted_df.iloc[i]['Type']
        
        # Set color by strategy
        bar.set_color(colors.get(strategy, 'gray'))
        
        # Set pattern by asset type
        if asset_type in hatches:
            bar.set_hatch(hatches[asset_type])
    
    # Add labels and title
    plt.title('Investment Returns by Asset and Strategy', fontsize=16)
    plt.xlabel('Asset - Strategy', fontsize=12)
    plt.ylabel('Return (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add a legend for strategies
    from matplotlib.patches import Patch
    strategy_elements = [Patch(facecolor=color, label=strategy) 
                       for strategy, color in colors.items() if strategy in results_df['Strategy'].values]
    
    # Add a legend for asset types
    type_elements = [Patch(facecolor='gray', hatch=hatch, label=asset_type) 
                   for asset_type, hatch in hatches.items() if asset_type in results_df['Type'].values]
    
    # Combine legends
    plt.legend(handles=strategy_elements + type_elements, loc='upper right', fontsize=10)
    
    # Add horizontal line at 0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        value = sorted_df.iloc[i]['Return']
        plt.text(i, value + (5 if value >= 0 else -15), 
                f"{value:.1f}%", ha='center')
    
    plt.tight_layout()
    
    # Save the figure first before showing
    plt.savefig('market_timing_returns.png', dpi=300, bbox_inches='tight')
    print("\nBar chart saved as 'market_timing_returns.png'")
    
    # Then try to show it (this works in environments with display capability)
    try:
        plt.show()
    except Exception as e:
        print(f"Note: Could not display plot due to: {e}")
        print("However, the chart has been saved to 'market_timing_returns.png'")

# Run the analysis
if __name__ == "__main__":
    # Define tickers to analyze - mix of stocks and cryptocurrencies
    # tickers = ['AAPL', 'SOXX', 'VTI', 'BTC-USD', 'ETH-USD']
    tickers = ['SOXX', 'BTC-USD']
    
    # Set years to analyze
    analysis_years = 10
    
    # Set investment amounts
    yearly_investment = 2000  # $2000 per year for strategies 1-3
    daily_investment = 20     # $20 per trading day for strategy 4
    
    # Set whether to invest on financial report day instead of January 1st
    invest_on_financial_day = True  # Set to True to invest day before financial reports
    
    fin_day_label = "day before financial report" if invest_on_financial_day else "January 1st"
    
    print("Starting unified market timing analysis...")
    print(f"Analyzing performance of {tickers} for the past {analysis_years} years")
    print(f"Strategy 1: ${yearly_investment} invested at yearly low")
    print(f"Strategy 2: ${yearly_investment} invested on {fin_day_label} (stocks only)")
    print(f"Strategy 3: ${yearly_investment} invested at yearly high")
    print(f"Strategy 4: ${daily_investment} invested daily")
    
    # Run analysis
    results = analyze_assets(tickers, years=analysis_years, 
                            annual_investment=yearly_investment, 
                            daily_investment=daily_investment,
                            invest_on_financial_day=invest_on_financial_day)
    
    # Display results table
    print("\nResults Summary:")
    pd.set_option('display.precision', 2)
    
    if results.empty:
        print("No results available to display")
    else:
        try:
            print(results.to_string(index=False))
        except Exception as e:
            print(f"Error displaying results: {e}")
            print("Raw results data:")
            print(results)
    
    # Plot results and save to CSV as backup
    try:
        plot_results(results)
    except Exception as e:
        print(f"Error creating plot: {e}")
        print("Attempting to save results to CSV instead...")
        try:
            results.to_csv('market_timing_results.csv', index=False)
            print("Results saved to 'market_timing_results.csv'")
        except Exception as csv_error:
            print(f"Error saving CSV: {csv_error}")
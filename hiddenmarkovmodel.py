import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
def load_and_preprocess_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Print the column names to check the structure
    print("Columns in the CSV file:", df.columns.tolist())
    
    # Check if 'Date' column exists, if not, create it from the index
    if 'Date' not in df.columns:
        df['Date'] = pd.to_datetime(df.index)
    else:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Set 'Date' as the index
    df.set_index('Date', inplace=True)
    
    # Ensure 'Close' and 'Volume' columns exist
    required_columns = ['Close', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the CSV file")
    
    # Calculate returns and volatility
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=24).std()
    
    # Calculate volume change
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    return df

# Train HMM
def train_hmm(data, n_components=3):
    # Prepare features
    features = ['Returns', 'Volatility', 'Volume_Change']
    X = data[features].values
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Train HMM
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", n_iter=100, random_state=42)
    model.fit(X_scaled)
    return model, scaler

# Predict states
def predict_states(model, data, scaler):
    features = ['Returns', 'Volatility', 'Volume_Change']
    X = data[features].values
    X_scaled = scaler.transform(X)
    states = model.predict(X_scaled)
    return states

# Analyze states
def analyze_states(data, states, model):
    df_analysis = data.copy()
    df_analysis['state'] = states
    for state in range(model.n_components_):
        state_data = df_analysis[df_analysis['state'] == state]
        print(f"State {state}:")
        print(state_data[['Returns', 'Volatility', 'Volume_Change']].describe())
        print("\n")

# Plot results
def plot_results(data, states, model):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    # Plot prices and states
    ax1.plot(data.index, data['Close'])
    ax1.set_title('Bitcoin Price and HMM States')
    ax1.set_ylabel('Price')
    # Color the background based on the state
    for state in range(model.n_components_):
        mask = (states == state)
        ax1.fill_between(data.index, data['Close'].min(), data['Close'].max(), where=mask, alpha=0.3, label=f'State {state}')
    ax1.legend()
    # Plot returns
    ax2.plot(data.index, data['Returns'])
    ax2.set_title('Bitcoin Returns')
    ax2.set_ylabel('Returns')
    ax2.set_xlabel('Date')
    plt.tight_layout()
    plt.show()

# Main execution
file_path = 'C:/Users/Mohamed/Desktop/CryptoBot-ThomasBalisticTrades/download/btcusd-h1-bid-2019-01-13-2019-01-14.csv'

try:
    data = load_and_preprocess_data(file_path)
    print("Data loaded successfully. Shape:", data.shape)
    print("First few rows of the data:")
    print(data.head())
    
    model, scaler = train_hmm(data)
    states = predict_states(model, data, scaler)
    analyze_states(data, states, model)
    plot_results(data, states, model)

    # Print transition matrix
    print("Transition Matrix:")
    print(model.transmat_)

    # Print means and covariances of each state
    print("\nMeans and Covariances of each state:")
    for i in range(model.n_components_):
        print(f"State {i}:")
        print("Mean:", model.means_[i])
        print("Covariance:", model.covars_[i])
        print()

except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("Please check your CSV file structure and ensure it contains the required columns.")
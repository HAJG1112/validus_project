import numpy as np
from scipy.optimize import brentq
import pandas as pd
import itertools
from typing import List, Union
import seaborn as sns
import matplotlib.pyplot as plt

# Function to generate all price paths
def generate_price_paths(S0: float, v: float, N: int) -> np.ndarray:
    """
    Generate all possible price paths under binomial model

    Arguments:
    S0 -- Starting Value of asset price
    v -- asset price change coefficient
    N -- number of steps in binomial model

    Returns:
    a -- array of all possible (2^N) price paths
    """
    paths = itertools.product([1-v, 1+v], repeat=N)  # technically with starting condition we need an N+1 period model
    a = np.array(list(paths))
    a = np.insert(a, 0, S0, axis=1)
    for i in range(a.shape[0]):  # loop through each path
        for j in range(N):
            a[i, j+1] = a[i, j+1] * a[i, j]
    return np.round(a, 5)  # return all price paths to 5dp due to floating point errors in multiplication

# Function to calculate European put option value
def european_put_option_value(S0: float, v: float, K: float, N: int) -> float:
    # Calculate up and down factors
    u = 1 + v
    d = 1 - v
    # Initialize arrays for stock prices and option values
    S = np.zeros((N+1, N+1))
    V = np.zeros((N+1, N+1))
    # Compute stock prices at expiration
    for j in range(N+1):
        S[N][j] = S0 * (u**(N-j)) * (d**j)
        # Compute option values at expiration (put option)
        V[N][j] = max(0, K - S[N][j])

    # Backward induction to compute option values at earlier periods
    for i in range(N-1, -1, -1):
        for j in range(i+1):
            V[i][j] = (1/2) * (V[i+1][j] + V[i+1][j+1])
    return V[0][0]

# Function to calibrate v given option value
def calibrate_v(S: float, K: float, V: float, N: int) -> Union[float, None]:
    # Define objective function
    def objective_function(v):
        return european_put_option_value(S, v, K, N) - V
    try:
        # Solve for v using Brent's method
        v = brentq(objective_function, 0, 1)
        return v
    except ValueError:
        print("Given option value cannot be achieved under the model assumptions.")
        return None

# Function to calculate expectation of maximum price of path
def max_asset_expectation(price_path: np.ndarray) -> float:
    return np.max(price_path, axis=1).sum() / price_path.shape[0]  # sum(max(S_n))/num of paths

# Function to calculate fair market value of European put option
def calculate_option_premium(S0: float, K: float, v: float, N: int, notional: float) -> float:
    option_value = european_put_option_value(S0, v, K, N)
    option_premium = option_value * notional
    return option_premium

# Function to convert GBP cashflows to USD and calculate IRR
def irr(cashflows: List[float], guess: float = 0.1) -> float:
    """
    Calculate the Internal Rate of Return (IRR) using Newton's method.

    Arguments:
    cashflows -- List of cash flows, where the initial investment is negative
    guess -- Initial guess for the IRR (default: 0.1)

    Returns:
    irr -- Internal Rate of Return (IRR)
    """
    epsilon = 1e-6  # Error tolerance
    max_iter = 1000  # Maximum number of iterations
    x0 = guess
    for i in range(max_iter):
        npv = sum([cf / (1 + x0) ** t for t, cf in enumerate(cashflows)])
        npv_prime = sum([-t * cf / (1 + x0) ** (t + 1) for t, cf in enumerate(cashflows)])
        x1 = x0 - npv / npv_prime
        if abs(x1 - x0) < epsilon:
            return x1
        x0 = x1
    raise ValueError("Unable to converge to a solution")

# calculate unhedged IRRs
def calculate_unhedged_irr(cashflows: np.ndarray, paths: np.ndarray) -> List[float]:
    irrs = []
    for i in range(len(paths)):
        # multiply every cashflow into the relevant USD at future spot and calculate IRR
        irrs.append(irr(paths[i][::2] * np.array(cashflows), 0.1))
    return irrs

# Function to calculate IRR of hedged portfolio
def calculate_hedged_irr(paths: np.ndarray, cashflows: np.ndarray, notional: float, premium: float) -> List[float]:
    irrs = []
    for i in range(len(paths)):
        # multiply every cashflow into the relevant USD at future spot and calculate IRR
        payoff = max(0, (K-paths[i][-1]) * notional)
        usd_cf = paths[i][::2] * np.array(cashflows)
        # modifying T=0 and T=10 Cashflow conditions for option payoff
        usd_cf[0] -= premium
        usd_cf[-1] += payoff
        irrs.append(irr(usd_cf, 0.1))
    return irrs

if __name__ == "__main__":

    from time import perf_counter
    start = perf_counter() 

    S0 = 1.28065
    v = 0.05
    K = S0
    N = 10
    notional = 100000000

    paths = generate_price_paths(S0, v, N)

    pd.DataFrame(paths).to_csv("GBPUSD_paths.csv")

    max_paths = round(max_asset_expectation(paths), 5)

    print(f"Q3: expectation of the maximum of GBPUSD price path: {max_paths}")

    cashflow = pd.read_excel("Quantitative_Analyst_Case_Study_2024_Cashflow_Model.xlsx")

    cf = cashflow["Cashflow Amount (in Local Asset Currecny)"]

    unhedged_irr = calculate_unhedged_irr(cf, paths)

    print("Q4: Saving IRR of paths and histogram")

    pd.DataFrame(unhedged_irr, columns=["IRRs"]).to_csv("unhedged_irr_paths.csv")

    sns.histplot(np.dot(unhedged_irr, 100), bins=10, kde=False, color='skyblue')

    # Add labels and title
    plt.xlabel('Returns %')
    plt.ylabel('Frequency')
    plt.title('Histogram of Unhedged IRRs')

    put_option_premium = calculate_option_premium(S0, K, v, N, notional)

    print(f"Q5: European Put Option Value {K}x£100k = £{round(put_option_premium, 2)}")

    print("Q6: Saving IRR of paths and histogram")

    hedged_irr = calculate_hedged_irr(paths, cf, notional, put_option_premium)

    pd.DataFrame(hedged_irr, columns=["IRRs"]).to_csv("hedged_irr_paths.csv")

    sns.histplot(np.dot(hedged_irr, 100), bins=10, kde=False, color='red')
    handles = [plt.Rectangle((0,0),1,1, color='skyblue'), plt.Rectangle((0,0),1,1, color='red')]
    labels = ['Unhedged IRRs', 'Hedged IRRs']

    # Add labels and title
    plt.xlabel('Returns %')
    plt.ylabel('Frequency')
    plt.title('IRRs of hedged and unhedged Investment')
    plt.legend(handles, labels)
    plt.savefig('IRR histograms.png')

    end = perf_counter()
    print(f"Time taken to run: {end-start}")
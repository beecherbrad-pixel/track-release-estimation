import pandas as pd
import numpy as np

from statsmodels.tsa.api import VAR

def prepare_state_timeseries(df, epsilon=1e-6, drop_zeros=False):

    ts = (
        df.groupby('date')[['x_t_daily', 'x_g_t_daily']]
        .first()
        .sort_index()
    )
    
    if drop_zeros:
        ts = ts[(ts['x_t_daily'] > 0) & (ts['x_g_t_daily'] > 0)]
    
    ts['log_x_t'] = np.log(ts['x_t_daily'] + epsilon)
    ts['log_x_g'] = np.log(ts['x_g_t_daily'] + epsilon)
    
    ts = ts[['log_x_t', 'log_x_g']]
    ts = ts.replace([np.inf, -np.inf], np.nan).dropna()
    
    return ts

def estimate_var1(ts_data):
 
    model = VAR(ts_data)
    results = model.fit(1)
    
    phi = results.coefs[0]
    sigma = results.sigma_u
    
    return phi, sigma, results

def check_stationarity(phi, verbose=True):
    eigenvalues = np.linalg.eigvals(phi)
    moduli = np.abs(eigenvalues)
    
    is_stationary = np.all(moduli < 1)
    
    if verbose:
        print("Eigenvalues:")
        for i, ev in enumerate(eigenvalues):
            print(f"Lambda {i+1}: {ev:.4f} (|λ| = {np.abs(ev):.4f})")
        
        if is_stationary:
            print("\nSystem is stationary")
        else:
            print("\nSystem is NOT stationary")
    
    return is_stationary, eigenvalues

def estimate_state_process(df, epsilon=1e-6, drop_zeros=False):

    print(f"Preparing state time series...")
    ts = prepare_state_timeseries(df, epsilon, drop_zeros)
    
    print(f"Estimating VAR(1)...")
    phi, sigma, results = estimate_var1(ts)
    
    print(f"Checking stationarity...")
    is_stationary, eigenvalues = check_stationarity(phi)
    
    return {
        'phi': phi,
        'sigma': sigma,
        'ts_data': ts,
        'is_stationary': is_stationary,
        'eigenvalues': eigenvalues,
        'results': results
    }
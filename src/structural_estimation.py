import pandas as pd
import numpy as np

import jax
import jax.numpy as jnp
import optax

def prepare_jax_data(est_df):
    all_artists = est_df['fe_group'].unique()
    artist_map = {name: i for i, name in enumerate(all_artists)}
    
    data = {
        'age': jnp.array(est_df['age_days'].values),
        'x_tau': jnp.array(est_df['x_tau_norm'].values),
        'x_g_tau': jnp.array(est_df['x_g_tau_norm'].values),
        'y': jnp.array(est_df['daily_streams'].values),
        'artist_idx': jnp.array(est_df['fe_group'].map(artist_map).values)
    }
    
    return data, artist_map

def init_params(num_artists):
    return {
        'beta_m': jnp.array(0.0),
        'beta_g': jnp.array(0.0),
        'log_lambda': jnp.log(jnp.array(5.0)),
        'artist_effects': jnp.ones((num_artists,)) * 0.1
    }

def model_fn(params, age, x_tau, x_g_tau, artist_idx):
    beta_m = jnp.exp(params['beta_m'])
    beta_g = jnp.exp(params['beta_g'])
    lambda_ = jnp.exp(params['log_lambda'])
    
    eta = params['artist_effects']
    artist_fe = eta[artist_idx]
    
    age_scaled = age / 365.0
    
    return artist_fe + beta_m * x_tau + beta_g * x_g_tau - lambda_ * age_scaled


def loss_fn(params, data, reg_weight=1e-2):
    preds = model_fn(
        params,
        data['age'],
        data['x_tau'],
        data['x_g_tau'],
        data['artist_idx']
    )
    
    log_y = jnp.log(data['y'] + 1e-8)
    
    mse = jnp.mean((preds - log_y)**2)
    
    reg = reg_weight * jnp.mean(params['artist_effects']**2)
    
    return mse + reg

def build_optimizer():
    optimizer = optax.multi_transform(
        {
            'fast': optax.adam(1e-2),
            'slow': optax.adam(1e-3),
            'fe': optax.adam(2e-3)
        },
        {
            'beta_m': 'fast',
            'beta_g': 'fast',
            'log_lambda': 'slow',
            'artist_effects': 'fe'
        }
    )
    return optimizer

optimizer = build_optimizer()

@jax.jit
def train_step(params, opt_state, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

def get_batches(data, batch_size, key):
    n = data['y'].shape[0]
    perm = jax.random.permutation(key, n)
    
    for i in range(0, n, batch_size):
        idx = perm[i:i+batch_size]
        yield {k: v[idx] for k, v in data.items()}

def train_model(data, num_artists, epochs=100, batch_size=4096, seed=42):
    
    key = jax.random.PRNGKey(seed)
    
    params = init_params(num_artists)
    optimizer = build_optimizer()
    opt_state = optimizer.init(params)
    
    for epoch in range(epochs):
        key, subkey = jax.random.split(key)
        
        for batch in get_batches(data, batch_size, subkey):
            params, opt_state, loss = train_step(params, opt_state, batch)
        
        
    print(f"Epoch {epoch+1}: Loss = {loss}")
    return params

def summarize_params(params):
    beta_m = float(jnp.exp(params['beta_m']))
    beta_g = float(jnp.exp(params['beta_g']))
    lambda_ = float(jnp.exp(params['log_lambda']))
    
    eta = np.array(params['artist_effects'])
    
    print(f"Market Beta: {beta_m:.4f}")
    print(f"Genre Beta:  {beta_g:.4f}")
    print(f"Lambda:      {lambda_:.4f}")
    print(f"Half-life:   {np.log(2)/lambda_:.2f} years")
    print(f"Mean Eta:    {eta.mean():.4f}")
    print(f"Eta Range:   [{eta.min():.2f}, {eta.max():.2f}]")
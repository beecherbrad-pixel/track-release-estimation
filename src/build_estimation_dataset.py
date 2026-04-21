import pandas as pd
import numpy as np
import ast


def prepare_base_dataset(charts_df, dataset_df):
    print(f"Preparing base df...")
    dataset_df = dataset_df.drop_duplicates(subset='track_id')
    
    df = pd.merge(charts_df, dataset_df, on='track_id', how='inner')
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    
    return df

def add_track_age(df):
    print(f"Adding approximate track age...")
    proxy_dates = (
        df.groupby('track_id')['date']
        .min()
        .reset_index()
        .rename(columns={'date': 'approx_release_date'})
    )
    
    df = df.merge(proxy_dates, on='track_id')
    df['age_days'] = (df['date'] - df['approx_release_date']).dt.days
    
    return df

def clean_artist_column(df, col='artists_x'):
    print(f"Cleaning the artist column...")
    def fix_artist_column(val):
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            val = val.strip()
            if val.startswith('[') and val.endswith(']'):
                try:
                    return ast.literal_eval(val)
                except:
                    return [a.strip("'\" ") for a in val[1:-1].split(',')]
            return [val]
        return []
    
    df[col] = df[col].apply(fix_artist_column)
    return df

def compute_artist_power(df, artist_col='artists_x'):
    print(f"Computing artist 'power'...")
    exploded = df.explode(artist_col)
    return exploded.groupby(artist_col)['streams'].mean().to_dict()

def assign_artist_groups(df, artist_power, top_k=50):
    print(f"Assignming artist groups as top {top_k} artists or other...")
    def get_alpha(artist_list):
        return max(artist_list, key=lambda x: artist_power.get(x, 0))
    
    df['dominant_artist'] = df['artists_x'].apply(get_alpha)
    
    top_artists = df['dominant_artist'].value_counts().nlargest(top_k).index
    
    df['fe_group'] = df['dominant_artist'].apply(
        lambda x: x if x in top_artists else 'Other'
    )
    
    dummies = pd.get_dummies(df['fe_group'], prefix='art', drop_first=True)
    
    return pd.concat([df, dummies], axis=1), dummies.columns.tolist()

def compute_daily_streams_simple(df):
    print(f"Interpolating daily streams...")
    df = df.sort_values(['fe_group', 'track_id', 'date', 'streams'])
    
    df = df.drop_duplicates(
        subset=['fe_group', 'track_id', 'date'], 
        keep='last'
    )
    
    df['delta_streams'] = df.groupby(['fe_group', 'track_id'])['streams'].diff()
    df['delta_days'] = df.groupby(['fe_group', 'track_id'])['date'].diff().dt.days
    
    df['daily_streams'] = df['delta_streams'] / df['delta_days']
    
    df['daily_streams'] = (
        df['daily_streams']
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
        .clip(lower=0)
    )
    
    return df.drop(columns=['delta_streams', 'delta_days'])

def add_market_features(df):

    # Generate daily market heat
    print(f"Generating daily market heat...")
    market_heat = (
        df.groupby('date')['daily_streams']
        .sum()
        .reset_index()
        .rename(columns={'daily_streams': 'x_t_daily'})
    )


    # Generate daily genre-specific market heat
    print(f"Generating daily genre-specific market heat...")
    genre_heat = (
        df.groupby(['date', 'track_genre'])['daily_streams']
        .sum()
        .reset_index()
        .rename(columns={'daily_streams': 'x_g_t_daily'})
    )

    df = df.merge(market_heat, on='date', how='inner')
    df = df.merge(genre_heat, on=['date', 'track_genre'], how='left')

    df['log_x_t_daily'] = np.log(df['x_t_daily'] + 1)
    df['log_x_g_t_daily'] = np.log(df['x_g_t_daily'] + 1)


    # Obtain age 0 conditions
    print(f"Obtaining age 0 conditions...")
    release_lookup = (
        df[df['age_days'] == 0][
            ['track_id', 'x_t_daily', 'x_g_t_daily']
        ]
        .drop_duplicates(subset=['track_id'])
        .rename(columns={
            'x_t_daily': 'x_tau_daily',
            'x_g_t_daily': 'x_g_tau_daily'
        })
    )

    df = df.merge(release_lookup, on='track_id', how='inner')

    df = df[
        (df['x_tau_daily'] > 0) &
        (df['x_g_tau_daily'] > 0) &
        (df['age_days'] >= 0)
    ]

    # Normalization
    print(f"Normalizing...")
    df['x_tau_norm'] = df['x_tau_daily'] / 1e6
    df['x_g_tau_norm'] = df['x_g_tau_daily'] / 1e4

    return df

def build_estimation_dataset(charts_df, dataset_df, top_k=50):
    print(f"Building the estimation dataset...")
    df = prepare_base_dataset(charts_df, dataset_df)
    df = add_track_age(df)
    
    df = clean_artist_column(df)
    artist_power = compute_artist_power(df)
    
    df, dummy_cols = assign_artist_groups(df, artist_power, top_k)
    
    df = compute_daily_streams_simple(df)

    df = add_market_features(df)

    print(f"Estimation dataset complete!")
    return df, dummy_cols
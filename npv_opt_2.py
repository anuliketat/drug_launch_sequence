import streamlit as st
import openpyxl
import pandas as pd
import numpy as np
import os
import json
from forex_python.converter import CurrencyRates
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
from datetime import date as dat
from dateutil.relativedelta import relativedelta
from collections import OrderedDict
import itertools
import random
import warnings
import requests
from PIL import Image
from io import BytesIO
import mpld3
import streamlit.components.v1 as components
from matplotlib.ticker import PercentFormatter
import time
import io
import leafmap.foliumap as leafmap

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("NPV Simulator")

def _calculate_irp(COUNTRY: str, launch_data: pd.DataFrame, pre_data: pd.DataFrame, n: int, base_df: pd.DataFrame, date_launch: str, cont_df: pd.DataFrame, is_at_launch: bool = True) -> dict:
    """Calculates IRP for a given country."""
    d = launch_data[launch_data['Country'] == COUNTRY].iloc[0]
    min_c = int(d['Min countries']) if isinstance(d['Min countries'], (int, float)) and not pd.isna(d['Min countries']) else 0
    BASKET = d['Primary Basket'] or d['Secondary Basket']
    BASKET = list(filter(None, BASKET)) if isinstance(BASKET, str) and BASKET != 'nan' else [COUNTRY]
    periodicity = int(d['periodicity']) if isinstance(d['periodicity'], (int, float)) and not pd.isna(d['periodicity']) else 12
    at_period = 12 if periodicity <= 12 else periodicity
    end_date = cont_df[cont_df['Date'] > date_launch].reset_index(drop=True).iloc[:at_period]['Date'].iloc[-1]
    date_range = cont_df[cont_df['Date'] <= end_date]['Date'].tolist()
    mul = float(d['Multiplier']) if isinstance(d['Multiplier'], (int, float)) and not pd.isna(d['Multiplier']) else 1.0
    irp_ref_month = int(d['IRP calculation month']) if isinstance(d['IRP calculation month'], (int, float)) and not pd.isna(d['IRP calculation month']) else 0

    price_dic = {}
    base_price = base_df[base_df['Country'] == COUNTRY]['Base Price'].iloc[0]

    for date in date_range:
        if date < date_launch:
            price_dic[date] = 0
            continue

        if not pd.isna(periodicity):
            if date <= date_range[periodicity - 1]:
                price_dic[date] = base_price
            else:
                x_dt = date_range.index(date) - irp_ref_month if irp_ref_month else date_range.index(date)
                bask = [i for i in BASKET if not pre_data[(pre_data['Country'] == i) & (pre_data['Date'] == date_range[x_dt])].empty]
                if len(bask) < min_c:
                    price_dic[date] = base_price
                else:
                    price_data = pre_data[(pre_data['Country'].isin(bask)) & (pre_data['Date'] == date_range[x_dt])]['Base Price']
                    if launch_data.name == "irp_at":
                        metric = launch_data['at_launch'].iloc[0]
                    else:
                        metric = launch_data['post_launch'].iloc[0]

                    if metric == 'Min':
                        price_dic[date] = price_data.min() * mul
                    elif metric == 'Avg':
                        price_dic[date] = price_data.sort_values().iloc[:n].mean() * mul
                        if COUNTRY == 'Denmark':
                            price_dic[date] = x_df[(x_df['Country'].isin(bask)) & (x_df['Base Price'] != 0)]['Base Price'].mean() * mul
                    elif metric == 'Free':
                        price_dic[date] = base_price * mul
        else:
            _d = list(dict(OrderedDict(sorted(at_dic.items(), reverse=True))).items())[0][1] if at_dic else base_price  # Handle empty at_dic
            price_dic[date] = _d * mul if date == date_range[0] else price_dic[date_range[date_range.index(date) - 1]]

    return price_dic


def IRP_AT(COUNTRY: str, at_launch: str, pre_data: pd.DataFrame, n: int, base_df: pd.DataFrame, date_launch: str, cont_df: pd.DataFrame) -> dict:
    """Calculates at-launch IRP."""
    global irp_at, x_df
    irp_at['at_launch'] = at_launch
    return _calculate_irp(COUNTRY, irp_at, pre_data, n, base_df, date_launch, cont_df, is_at_launch=True)


def IRP_POST(COUNTRY: str, post_launch: str, at_dic: dict, pre_data: pd.DataFrame, n: int, cont_df: pd.DataFrame) -> dict:
    """Calculates post-launch IRP."""
    global irp_post
    irp_post['post_launch'] = post_launch
    launch_date = cont_df[cont_df['Date']>max(list(at_dic.keys()))]['Date'].iloc[0] if at_dic else cont_df['Date'].iloc[0] # Handle empty at_dic
    return _calculate_irp(COUNTRY, irp_post, pre_data, n, base_df, launch_date, cont_df, is_at_launch=False)


def _irp_temp(COUNTRY: str, at_metric: str, post_metric: str, n: int, new_data: pd.DataFrame, base_df: pd.DataFrame, date_launch: str, cont_df: pd.DataFrame) -> pd.DataFrame:
    """Combines at-launch and post-launch IRP."""
    dd_dic_at = IRP_AT(COUNTRY, at_metric, new_data, n, base_df, date_launch, cont_df)
    max_date = max(dd_dic_at.keys())
    dd_at = cont_df[cont_df['Date'] <= max_date].copy()
    dd_at['Base Price'] = dd_at['Date'].map(dd_dic_at)

    dd_dic_post = IRP_POST(COUNTRY, post_metric, dd_dic_at, new_data, n, cont_df)
    dd_post = cont_df[cont_df['Date'] > max_date].copy()
    dd_post['Base Price'] = dd_post['Date'].map(dd_dic_post)

    dd_full = pd.concat([dd_at, dd_post], ignore_index=True)
    return dd_full


def cal_data(base_long: pd.DataFrame, act_df: pd.DataFrame, upd_price_df: pd.DataFrame) -> pd.DataFrame:
    """Computes COGS, volumes, discounts, and clawbacks."""
    vol_df = []
    for c in upd_price_df['Country'].unique():
        start_date = base_long[(base_long['Country'] == c) & (base_long['Base Price'] != 0)]['Date'].iloc[0] if not base_long[(base_long['Country'] == c) & (base_long['Base Price'] != 0)].empty else None # Handle potential empty series
        if start_date is None: # If start date is not found skip the country
            continue
        vol_c = act_df[act_df['Country'] == c].copy()
        vol_c.columns = [str(i).split()[0] for i in vol_c.columns]

        first_non_zero_col_index = np.where(vol_c.values != 0)[1].min() if np.any(vol_c.values != 0) else None
        if first_non_zero_col_index is not None:
            vol_c_date = vol_c.columns[first_non_zero_col_index]
            first_non_zero_val = vol_c.values[np.where(vol_c.values != 0)][0]
            if start_date <= vol_c_date:
                cols_to_replace = vol_c.columns[vol_c.columns.get_loc(start_date):vol_c.columns.get_loc(vol_c_date)]
                vol_c.loc[:, cols_to_replace] = vol_c.loc[:, cols_to_replace].replace(0, first_non_zero_val)
            else:
                cols_to_zero = vol_c.columns[vol_c.columns.get_loc(vol_c_date):vol_c.columns.get_loc(start_date)]
                vol_c.loc[:, cols_to_zero] = 0
        vol_df.append(vol_c)

    return pd.concat(vol_df, ignore_index=True)


def NPV(price_df: pd.DataFrame, cogs_df: pd.DataFrame, vol_df: pd.DataFrame, dis_df: pd.DataFrame, claw_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Net Present Value (NPV)."""
    wacc = 0.075
    npv_results = []

    for country in price_df['Country'].unique():
        p = price_df[price_df['Country'] == country].iloc[0, 1:].values.astype(float) if not price_df[price_df['Country'] == country].empty else None # Handle potential empty series
        c = cogs_df[cogs_df['Country'] == country].iloc[0, 1:].values.astype(float) if not cogs_df[cogs_df['Country'] == country].empty else None
        v = vol_df[vol_df['Country'] == country].iloc[0, 1:].values.astype(float) if not vol_df[vol_df['Country'] == country].empty else None
        d = dis_df[dis_df['Country'] == country].iloc[0, 1:].values.astype(float) if not dis_df[dis_df['Country'] == country].empty else None
        cl = claw_df[claw_df['Country'] == country].iloc[0, 1:].values.astype(float) if not claw_df[claw_df['Country'] == country].empty else None

        if any(x is None for x in [p,c,v,d,cl]): # If any of the dataframes are empty for a country skip it
            continue

        profit = p - c - d
        revenue = (profit * v) - cl

        periods = np.arange(1, 11)
        discount_factors = (1 + wacc)**periods
        discounted_revenue = revenue.reshape(-1, 12) / discount_factors.reshape(1, -1)
        npv = np.sum(discounted_revenue, axis=1)

        npv_df = pd.DataFrame({'Country': country, 'NPV': npv})
        npv_results.append(npv_df)

    return pd.concat(npv_results, ignore_index=True)


def IRP(base_long: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Calculates IRP for all countries."""
    free_countries = ['France', 'Germany', 'United Kingdom', 'Sweden']
    irp_data = []

    for c in base['Country']:
        dx = base_long[base_long['Country'] == c].reset_index(drop=True)
        launch_date = dx[dx['Base Price'] != 0]['Date'].iloc[0] if not dx[dx['Base Price'] != 0].empty else None
        if launch_date is None:
            continue

        if c in free_countries:
            at_period = 12
            d_at = dx[:at_period].copy()
            d_post = dx[at_period:].copy()
            d_at['Base Price'] = np.where((d_at['Base Price'] == 0) & (d_at['Date'] > launch_date), d_at['Base Price'].max(), d_at['Base Price'])
            d_post['Base Price'] = d_at['Base Price'].max()
            dd_full = pd.concat([d_at, d_post], ignore_index=True)
        elif c in ('Belgium', 'Italy'):
            dd_full = dx.copy()
            dd_full['Base Price'] = np.where(dd_full['Date'] > launch_date, dx['Base Price'].max(), dd_full['Base Price'])
        elif c == 'Luxembourg':
            dd_full = dx[dx['Date'] <= launch_date].copy()
            temp = dx[dx['Date'] > launch_date].copy()
            belgium_prices = dfs[(dfs['Country'] == 'Belgium') & (dfs['Date'] > launch_date)]['Base Price'].tolist() if 'Belgium' in dfs['Country'].unique() else []
            temp['Base Price'] = belgium_prices if len(belgium_prices) > 0 else 0
            dd_full = pd.concat([dd_full, temp], ignore_index=True)
        else:
            at_metric, post_metric, n = _get_country_parameters(c, dfs)
            dd_full = _irp_temp(c, at_metric, post_metric, new_data=irp_data, n=n, base_df=base, date_launch=launch_date, cont_df=dx)

        dd_full['Country'] = c

        irp_data.append(dd_full)

    return pd.concat(irp_data, ignore_index=True)


def _get_country_parameters(country: str, dfs: pd.DataFrame) -> tuple:
    """Helper function to return the parameters for a country."""
    country_params = {
        'Cyprus': ('Avg', 'Avg', len(dfs) if not dfs.empty else 0),  # Handle empty dfs
        'Slovakia': ('Avg', 'Avg', 3),
        'Austria': ('Free', 'Avg', len(dfs) if not dfs.empty else 0),
        'Estonia': ('Min', 'Min', len(dfs) if not dfs.empty else 0),
        'Bulgaria': ('Min', 'Free', len(dfs) if not dfs.empty else 0),
        'Greece': ('Avg', 'Avg', 2),
        'Romania': ('Min', 'Min', len(dfs) if not dfs.empty else 0),
        'Czech Republic': ('Avg', 'Avg', 3),
        'Iceland': ('Avg', 'Avg', len(dfs) if not dfs.empty else 0),
        'Denmark': ('Avg', 'Avg', len(dfs) if not dfs.empty else 0),
        'Spain': ('Free', 'Min', len(dfs) if not dfs.empty else 0),
        'Hungary': ('Min', 'Min', len(dfs) if not dfs.empty else 0),
        'Ireland': ('Free', 'Avg', len(dfs) if not dfs.empty else 0),
        'Malta': ('Min', 'Min', len(dfs) if not dfs.empty else 0),
        'Poland': ('Min', 'Min', len(dfs) if not dfs.empty else 0),
        'Portugal': ('Min', 'Min', len(dfs) if not dfs.empty else 0),
        'Netherlands': ('Free', 'Avg', len(dfs) if not dfs.empty else 0),
        'Latvia': ('Free', 'Free', len(dfs) if not dfs.empty else 0),
        'Slovenia': ('Min', 'Min', len(dfs) if not dfs.empty else 0),
        'Switzerland': ('Avg', 'Avg', len(dfs) if not dfs.empty else 0),
        'Croatia': ('Avg', 'Avg', len(dfs) if not dfs.empty else 0),
        'Lithuania': ('Avg', 'Avg', 3),
        'Norway': ('Avg', 'Avg', 3),
        'Finland': ('Avg', 'Avg', 3),
    }
    return country_params.get(country, ('Avg', 'Avg', 0))  # Default to 0 if dfs is empty


def update_launch(npv_df: pd.DataFrame, irp_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Updates the launch sequence."""
    npv_df['npv'] = npv_df.iloc[:, 1:].sum(1)

    launch_cnts = irp_df['Launch Month'].value_counts().reset_index()
    max_val = launch_cnts[launch_cnts['Launch Month'] > n]['index'].values[0] if not launch_cnts[launch_cnts['Launch Month'] > n].empty else None #Handle the case where there are no months with launches more than N
    if max_val is None: # If there are no months with launches more than N then no update is needed
        return irp_df

    conts = irp_df[irp_df['Launch Month'] == max_val]['Country'].unique()

    d = npv_df[npv_df['Country'].isin(conts)].sort_values(['npv'], ascending=False)[:n]
    rem = list(set(conts) - set(d['Country'].tolist()))
    dic = {}
    for c in rem:
        new_val = max_val + 1
        c_range = irp_df[irp_df['Country'] == c]['range'].values[0]
        dic[c] = new_val if new_val <= max(c_range) else max(c_range)
        if dic[c] == max_val:
            dic[c] = dic[c] + 1

    new_irp = irp_df.copy()
    new_irp['Launch Month'] = new_irp['Country'].map(dic)
    new_irp['Launch Month'] = np.where(new_irp['Launch Month'].isnull(), irp_df['Launch Month'], new_irp['Launch Month'])
    new_irp['Launch Month'] = new_irp['Launch Month'].astype(int)
    return new_irp


# ... (rest of the Streamlit code)


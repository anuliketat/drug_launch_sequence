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
from collections import OrderedDict, Counter
import itertools
import random
import warnings
import mpld3
import streamlit.components.v1 as components
from matplotlib.ticker import PercentFormatter
import time
import io
import leafmap.foliumap as leafmap

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("NPV Simulator")
# Runs all functions and calculates NPV for a given sequence
def IRP_AT(COUNTRY, at_launch, pre_data, n, base_df, date_launch, cont_df):
  global irp_at, x_df
  # '''
  #   COUNTRY: Country
  #   at_launch: Metric - 'Avg'/'Min'/'Max'/'Free'
  #   pre_data: irp prices data for countries launched before
  #   n: n-lowest value
  #   base_df: IRP base data
  #   date_launch:
  # '''
  d = irp_at[irp_at['Country']==COUNTRY]
  min_c = d['Min countries'].values[0] if isinstance(d['Min countries'].values[0], int) else 0
  BASKET = d['Primary Basket'].values[0] if d['Primary Basket'] is not None else d['Secondary Basket'].values[0]
  BASKET = list(filter(None, BASKET)) if str(BASKET)!='nan' else [COUNTRY]
  periodicity = d['periodicity'].values[0]
  at_period = 12 if (periodicity<=12) or (isinstance(periodicity, float)) else periodicity
  end_date = cont_df[(cont_df['Date']>date_launch)].reset_index(drop=True)[:at_period]['Date'].values[-1]
  date_range = cont_df[(cont_df['Date']<=end_date)]['Date'].tolist()
  mul = d['Multiplier'].values[0]
  mul = 1 if str(mul)=='nan' else mul
  periodicity = d['periodicity'].values[0]
  irp_ref_month_at = d['IRP calculation month'].values[0] if isinstance(d['IRP calculation month'].values[0], int) else 0
  price_dic = {}
  for ix, date in enumerate(date_range):
    if date < date_launch:
      price_dic[date] = 0
    elif str(periodicity)!='nan':
      if date <= date_range[periodicity-1]:
        price_dic[date] = base_df[base_df['Country']==COUNTRY]['Base Price'].values[0]
      elif (date > date_range[periodicity-1]):
        x_dt = date_range.index(date)-irp_ref_month_at if str(irp_ref_month_at)!='nan' else date_range.index(date)
        bask = [i for i in BASKET if len(pre_data[(pre_data['Country']==i)&(pre_data['Date']==date_range[x_dt])])!=0]
        if (len(bask) < min_c):
          price_dic[date] = base_df[base_df['Country']==COUNTRY]['Base Price'].values[0]
        elif at_launch=='Min':
          price_dic[date] = pre_data[(pre_data['Country'].isin(bask))&(pre_data['Date']==date_range[x_dt])]['Base Price'].min() * mul
        elif at_launch=='Avg':
          price_dic[date] = pre_data[(pre_data['Country'].isin(bask))&(pre_data['Date']==date_range[x_dt])].sort_values(['Base Price'])[:n]['Base Price'].mean() * mul
          if COUNTRY=='Denmark':
            price_dic[date] = x_df[(x_df['Country'].isin(bask))&(x_df['Base Price']!=0)]['Base Price'].mean() * mul
        elif at_launch=='Free':
          price_dic[date] = base_df[base_df['Country']==COUNTRY]['Base Price'].values[0] * mul
    else:
      price_dic[date] = base_df[base_df['Country']==COUNTRY]['Base Price'].values[0]
  return price_dic

#@title Post Launch
def IRP_POST(COUNTRY, post_launch, at_dic, pre_data, n, cont_df):
  global irp_post
  d2 = irp_post[irp_post['Country']==COUNTRY]
  min_c_d2 = d2['Min countries'].values[0] if isinstance(d2['Min countries'].values[0], int) else 0
  BASKET_d2 = d2['Primary Basket'].values[0] if d2['Primary Basket'] is not None else d2['Secondary Basket'].values[0]
  BASKET_d2 = list(filter(None, BASKET_d2)) if str(BASKET_d2)!='nan' else [COUNTRY]

  date_range = cont_df[cont_df['Date']>max(list(at_dic.keys()))]['Date'].tolist()
  mul_post = d2['Multiplier'].values[0]
  mul_post = 1 if str(mul_post)=='nan' or isinstance(mul_post, str) else mul_post
  periodicity_post = d2['periodicity'].values[0]
  irp_ref_month_post = d2['IRP calculation month'].values[0] if isinstance(d2['IRP calculation month'].values[0], int) else 0
  price_dic = {}
  for ix, date in enumerate(date_range):
    if str(periodicity_post)!='nan':
      if (((ix+1) % periodicity_post==0) or ix==0):
        x_dt = date_range.index(date)-irp_ref_month_post if str(irp_ref_month_post)!='nan' else date_range.index(date)
        bask = [i for i in BASKET_d2 if len(pre_data[(pre_data['Country']==i)&(pre_data['Date']==date_range[x_dt])])!=0]
        if (len(bask) < min_c_d2):
          _d = list(dict(OrderedDict(sorted(at_dic.items(), reverse=True))).items())[0][1]
          price_dic[date] = _d * mul_post
        elif post_launch=='Min':
          price_dic[date] = pre_data[(pre_data['Country'].isin(bask))&(pre_data['Date']==date_range[x_dt])]['Base Price'].min() * mul_post
        elif post_launch=='Avg':
          price_dic[date] = pre_data[(pre_data['Country'].isin(bask))&(pre_data['Date']==date_range[x_dt])].sort_values(['Base Price'])[:n]['Base Price'].mean() * mul_post
          if COUNTRY=='Iceland':
            p = pre_data[(pre_data['Country'].isin(bask))&(pre_data['Date']==date_range[x_dt])].sort_values(['Base Price'])[:n]['Base Price'].mean() * mul_post
            annual = sum([price_dic[x] for x in list(price_dic)[-12:]])
            cr = CurrencyRates()
            converted_price = cr.convert(base_cur='EUR', dest_cur='ISK', amount=annual)
            if converted_price < 6e+06:
              markup = 0.15
              price_dic[date] = p*(1+markup)
            price_dic[date] = p
          elif COUNTRY=='Denmark':
            price_dic[date] = x_df[(x_df['Country'].isin(bask))&(x_df['Base Price']!=0)]['Base Price'].mean() * mul_post
          elif COUNTRY=='Greece':
            p = pre_data[(pre_data['Country'].isin(bask))&(pre_data['Date']==date_range[x_dt])].sort_values(['Base Price'])[:n]['Base Price'].mean() * mul_post
            cut_off = 0.7
            new_p = p*(1-cut_off)
            price_dic[date] = max(p, new_p)
        elif post_launch=='Free':
          _d = list(dict(OrderedDict(sorted(at_dic.items(), reverse=True))).items())[0][1]
          dx_val = date_range[date_range.index(date)-1]
          price_dic[date] = price_dic[dx_val] if date!=date_range[0] else _d * mul_post
      else:
        dx_val = date_range[date_range.index(date)-1]
        price_dic[date] = price_dic[dx_val]
    else:
      pre_val = date_range[date_range.index(date)-1] if date_range.index(date)!=0 else date_range[date_range.index(date)]
      _d = list(dict(OrderedDict(sorted(at_dic.items(), reverse=True))).items())[0][1]
      price_dic[date] = price_dic[pre_val] if date!=date_range[0] else _d * mul_post

  return price_dic

def _irp_temp(COUNTRY, at_metric, post_metric, n, new_data, base_df, date_launch, cont_df):
      # print(COUNTRY.upper(), 'At Launch')
  dd_dic = IRP_AT(COUNTRY, at_metric, new_data, n, base_df, date_launch, cont_df)
  max_date = max(list(dd_dic.keys()))
  dd_at = cont_df[cont_df['Date']<=max_date]
  dd_at['Base Price'] = dd_at['Date'].map(dd_dic)

  # print(COUNTRY.upper(), 'Post Launch')
  dd_dic_post = IRP_POST(COUNTRY, post_metric, dd_dic, new_data, n, cont_df)
  dd_post = cont_df[cont_df['Date']>max_date]
  dd_post['Base Price'] = dd_post['Date'].map(dd_dic_post)
  dd_full = pd.concat([dd_at, dd_post])
  return dd_full

#@title Cogs, Vol, Discounts, Clawback auto compute as per IRP
def cal_data(base_long, act_df, upd_price_df):
  # '''
  #   base_long: IRP Base long form data
  #   act_df: Base vol/cogs/dis/clawback data
  #   upd_price_df: irp updated prices data
  # '''
  vol_df = pd.DataFrame()
  for c in upd_price_df['Country'].unique():
    start_date = base_long[(base_long['Country']==c)&(base_long['Base Price']!=0)]['Date'].values[0]
    vol_c = act_df[act_df['Country']==c]
    vol_c.columns = [str(i).split()[0] for i in vol_c.columns]
    vol_c_date = next(vol_c.columns[ix+1] for ix, i in enumerate(vol_c.values.flatten()[1:]) if i!=0)
    if start_date<=vol_c_date:
      vol_c_val = next(i for ix,i in enumerate(vol_c.values.flatten()) if (not isinstance(i, str) and i!=0))
      for col in vol_c.columns[vol_c.columns.tolist().index(start_date):vol_c.columns.tolist().index(vol_c_date)]:
        vol_c[col].replace({0:vol_c_val}, inplace=True)
    else:
      for col in vol_c.columns[vol_c.columns.tolist().index(vol_c_date):vol_c.columns.tolist().index(start_date)]:
        vol_c[col] = 0
    vol_df = vol_df.append(vol_c, ignore_index=True)
  return vol_df

#@title NPV Function
def NPV(price_df, cogs_df, vol_df, dis_df, claw_df):
  pds = []
  for con in price_df['Country'].unique():
    p = price_df[price_df['Country']==con]
    c = cogs_df[cogs_df['Country']==con]
    v = vol_df[vol_df['Country']==con]
    d = dis_df[dis_df['Country']==con]
    cl = claw_df[claw_df['Country']==con]

    profit = p.values[:, 1:] - c.values[:, 1:] - d.values[:, 1:]
    rev = (profit * v.values[:, 1:]) - cl.values[:, 1:]
    rev = pd.DataFrame(rev, columns=p.columns[1:])
    rev['Country'] = con
    pds.append(rev)
  rev = pd.concat(pds)

  df = rev.drop(['Country'], axis=1)
  wacc = 0.075
  x, t = (0, 1)
  dfs = []
  while t<=10:
    ds = df.iloc[:, x:x+12]
    fac = (1 + wacc)**t
    ds /= fac
    dfs.append(ds)
    x += 12
    if x%12==0:
      t += 1

  y = pd.concat(dfs, axis=1)
  y['Country'] = rev['Country']
  cols = ['Country'] + y.columns[:-1].tolist()
  y = y[cols]
  return y


def IRP(base_long: pd.DataFrame, base: pd.DataFrame):
  free_countries = ['France', 'Germany', 'United Kingdom', 'Sweden']
  dfs = pd.DataFrame()
  for c in base['Country']:
    # print(c)
    dx = base_long[base_long['Country']==c].reset_index(drop=True)
    at_period = 12
    launch_date = dx[dx['Base Price']!=0]['Date'].values[0]
    if c in free_countries:
      d_at = dx[:at_period]
      d_post = dx[at_period:]
      d_at['Base Price'] = np.where((d_at['Base Price']==0)&(d_at['Date']>launch_date), d_at['Base Price'].max(), d_at['Base Price'])
      d_post['Base Price'] = d_at['Base Price'].max()
      d_full = pd.concat([d_at, d_post])
      dfs = dfs.append(d_full, ignore_index=True)
    elif c=='Cyprus':
      dd_full = _irp_temp(c, at_metric='Avg', post_metric='Avg', new_data=dfs, n=len(dfs), base_df=base, date_launch=launch_date, cont_df=dx)
      dd_full['Country'] = c
      dfs = dfs.append(dd_full, ignore_index=True)
    elif c=='Slovakia':
      dd_full = _irp_temp(c, at_metric='Avg', post_metric='Avg', new_data=dfs, n=3, base_df=base, date_launch=launch_date, cont_df=dx)
      dd_full['Country'] = c
      dfs = dfs.append(dd_full, ignore_index=True)
    elif c=='Austria':
      dd_full = _irp_temp(c, at_metric='Free', post_metric='Avg', new_data=dfs, n=len(dfs), base_df=base, date_launch=launch_date, cont_df=dx)
      dd_full['Country'] = c
      dfs = dfs.append(dd_full, ignore_index=True)
    elif c=='Estonia':
      dd_full = _irp_temp(c, at_metric='Min', post_metric='Min', new_data=dfs, n=len(dfs), base_df=base, date_launch=launch_date, cont_df=dx)
      dd_full['Country'] = c
      dfs = dfs.append(dd_full, ignore_index=True)
    elif c=='Bulgaria':
      dd_full = _irp_temp(c, at_metric='Min', post_metric='Free', new_data=dfs, n=len(dfs), base_df=base, date_launch=launch_date, cont_df=dx)
      dd_full['Country'] = c
      dfs = dfs.append(dd_full, ignore_index=True)
    elif c=='Greece':
      dd_full = _irp_temp(c, at_metric='Avg', post_metric='Avg', new_data=dfs, n=2, base_df=base, date_launch=launch_date, cont_df=dx)
      dd_full['Country'] = c
      dfs = dfs.append(dd_full, ignore_index=True)
    elif c=='Romania':
      dd_full = _irp_temp(c, at_metric='Min', post_metric='Min', new_data=dfs, n=len(dfs), base_df=base, date_launch=launch_date, cont_df=dx)
      dd_full['Country'] = c
      dfs = dfs.append(dd_full, ignore_index=True)
    elif c=='Czech Republic':
      dd_full = _irp_temp(c, at_metric='Avg', post_metric='Avg', new_data=dfs, n=3, base_df=base, date_launch=launch_date, cont_df=dx)
      dd_full['Country'] = c
      dfs = dfs.append(dd_full, ignore_index=True)
    elif c=='Iceland':
      dd_full = _irp_temp(c, at_metric='Avg', post_metric='Avg', new_data=dfs, n=len(dfs), base_df=base, date_launch=launch_date, cont_df=dx)
      dd_full['Country'] = c
      dfs = dfs.append(dd_full, ignore_index=True)
    elif c=='Denmark':
      dd_full = _irp_temp(c, at_metric='Avg', post_metric='Avg', new_data=dfs, n=len(dfs), base_df=base, date_launch=launch_date, cont_df=dx)
      dd_full['Country'] = c
      dfs = dfs.append(dd_full, ignore_index=True)
    elif c=='Belgium':
      dd_full = dx
      dd_full['Base Price'] = np.where((dd_full['Date']>launch_date), dx['Base Price'].max(), dd_full['Base Price'])
      dd_full['Country'] = c
      dfs = dfs.append(dd_full, ignore_index=True)
    elif c=='Spain':
      dd_full = _irp_temp(c, at_metric='Free', post_metric='Min', new_data=dfs, n=len(dfs), base_df=base, date_launch=launch_date, cont_df=dx)
      dd_full['Country'] = c
      dfs = dfs.append(dd_full, ignore_index=True)
    elif c=='Italy':
      dd_full = dx
      dd_full['Base Price'] = np.where((dd_full['Date']>launch_date), dx['Base Price'].max(), dd_full['Base Price'])
      dd_full['Country'] = c
      dfs = dfs.append(dd_full, ignore_index=True)
    elif c=='Hungary':
      dd_full = _irp_temp(c, at_metric='Min', post_metric='Min', new_data=dfs, n=len(dfs), base_df=base, date_launch=launch_date, cont_df=dx)
      dd_full['Country'] = c
      dfs = dfs.append(dd_full, ignore_index=True)
    elif c=='Ireland':
      dd_full = _irp_temp(c, at_metric='Free', post_metric='Avg', new_data=dfs, n=len(dfs), base_df=base, date_launch=launch_date, cont_df=dx)
      dd_full['Country'] = c
      dfs = dfs.append(dd_full, ignore_index=True)
    elif c=='Malta':
      dd_full = _irp_temp(c, at_metric='Min', post_metric='Min', new_data=dfs, n=len(dfs), base_df=base, date_launch=launch_date, cont_df=dx)
      dd_full['Country'] = c
      dfs = dfs.append(dd_full, ignore_index=True)
    elif c=='Poland': # Edit for ref month
      dd_full = _irp_temp(c, at_metric='Min', post_metric='Min', new_data=dfs, n=len(dfs), base_df=base, date_launch=launch_date, cont_df=dx)
      dd_full['Country'] = c
      dfs = dfs.append(dd_full, ignore_index=True)
    elif c=='Portugal':
      dd_full = _irp_temp(c, at_metric='Min', post_metric='Min', new_data=dfs, n=len(dfs), base_df=base, date_launch=launch_date, cont_df=dx)
      dd_full['Country'] = c
      dfs = dfs.append(dd_full, ignore_index=True)
    elif c=='Netherlands':
      dd_full = _irp_temp(c, at_metric='Free', post_metric='Avg', new_data=dfs, n=len(dfs), base_df=base, date_launch=launch_date, cont_df=dx)
      dd_full['Country'] = c
      dfs = dfs.append(dd_full, ignore_index=True)
    elif c=='Latvia':
      dd_full = _irp_temp(c, at_metric='Free', post_metric='Free', new_data=dfs, n=len(dfs), base_df=base, date_launch=launch_date, cont_df=dx)
      dd_full['Country'] = c
      dfs = dfs.append(dd_full, ignore_index=True)
    elif c=='Slovenia':
      dd_full = _irp_temp(c, at_metric='Min', post_metric='Min', new_data=dfs, n=len(dfs), base_df=base, date_launch=launch_date, cont_df=dx)
      dd_full['Country'] = c
      dfs = dfs.append(dd_full, ignore_index=True)
    elif c=='Switzerland':
      dd_full = _irp_temp(c, at_metric='Avg', post_metric='Avg', new_data=dfs, n=len(dfs), base_df=base, date_launch=launch_date, cont_df=dx)
      dd_full['Country'] = c
      dfs = dfs.append(dd_full, ignore_index=True)
    elif c=='Luxembourg':
      dd_full = dx[dx['Date']<=launch_date]
      temp = dx[dx['Date']>launch_date]
      temp['Base Price'] = dfs[(dfs['Country']=='Belgium')&(dfs['Date']>launch_date)]['Base Price'].tolist()
      dd_full = pd.concat([dd_full, temp])
      dd_full['Country'] = c
      dfs = dfs.append(dd_full, ignore_index=True)
    elif c=='Croatia':
      dd_full = _irp_temp(c, at_metric='Avg', post_metric='Avg', new_data=dfs, n=len(dfs), base_df=base, date_launch=launch_date, cont_df=dx)
      dd_full['Country'] = c
      dfs = dfs.append(dd_full, ignore_index=True)
    elif c=='Lithuania':
      dd_full = _irp_temp(c, at_metric='Avg', post_metric='Avg', new_data=dfs, n=3, base_df=base, date_launch=launch_date, cont_df=dx)
      dd_full['Country'] = c
      dfs = dfs.append(dd_full, ignore_index=True)
    elif c=='Norway':
      dd_full = _irp_temp(c, at_metric='Avg', post_metric='Avg', new_data=dfs, n=3, base_df=base, date_launch=launch_date, cont_df=dx)
      dd_full['Country'] = c
      dfs = dfs.append(dd_full, ignore_index=True)
    elif c=='Finland':
      dd_full = _irp_temp(c, at_metric='Avg', post_metric='Avg', new_data=dfs, n=3, base_df=base, date_launch=launch_date, cont_df=dx)
      dd_full['Country'] = c
      dfs = dfs.append(dd_full, ignore_index=True)

  return dfs


def update_launch(npv_df, irp_df, n):
  # '''
  #   npv_df: npv data
  #   irp_df: irp base data
  #   n: num countries constraint
  #   ------------------------------
  #   returns updated launch sequence data
  # '''
  npv_df['npv'] = npv_df.iloc[:, 1:].sum(1)

  launch_cnts = irp_df['Launch Month'].value_counts().reset_index()
  max_val = launch_cnts[launch_cnts['Launch Month']>n]['index'].values[0]
  print(max_val)
  conts = irp_df[irp_df['Launch Month']==max_val]['Country'].unique()

  d = npv_df[npv_df['Country'].isin(conts)].sort_values(['npv'], ascending=False)[:n]
  rem = list(set(conts) - set(d['Country'].tolist()))
  dic = {}
  for c in rem:
    new_val = max_val + 1
    c_range = irp_df[irp_df['Country']==c]['range'].values[0]
    dic[c] = new_val if new_val<=max(c_range) else max(c_range)
    if dic[c]==max_val:
      dic[c] = dic[c]+1

  new_irp = irp_df.copy()
  new_irp['Launch Month'] = new_irp['Country'].map(dic)
  new_irp['Launch Month'] = np.where(new_irp['Launch Month'].isnull(), irp_df['Launch Month'], new_irp['Launch Month'])
  new_irp['Launch Month'] = new_irp['Launch Month'].astype(int)
  return new_irp

label = """
        Upload Data in Excel with sheets for
        - Price, Volumes, Cogs, Discounts, Clawback for all markets.
        - IRP data with At Launch, Post Launch and Base sheets for all markets.
        """
st.markdown(label)
data_file = st.file_uploader(label="Upload", type=["xlsx"], label_visibility='collapsed')
if data_file is None:
    st.write('Please upload file!')
else:
    global irp_base
    price = pd.read_excel(data_file, sheet_name='Price')
    vol = pd.read_excel(data_file, sheet_name='Volume')
    cogs = pd.read_excel(data_file, sheet_name='Cogs')
    dis = pd.read_excel(data_file, sheet_name='Discount')
    claw = pd.read_excel(data_file, sheet_name='Clawback')
    cogs_seas = pd.read_excel(data_file, sheet_name='Cogs_Seasonality')
    cogs_seas.rename(columns={'Countries':'Country'}, inplace=True)
    for d in [price, vol, cogs, dis, claw]:
        for c in d.columns:
            d[c] = d[c].fillna(0)
    yrs = 10
    x = pd.date_range(start='2023-01-01', periods=365*yrs, freq='D')
    x = [str(i).split()[0] for i in x if str(i).split()[0].endswith('01')]
    for d in [price, vol, cogs, dis, claw]:
        d.columns = ['Country'] + x

    irp_at = pd.read_excel(data_file, sheet_name='At Launch')
    irp_post = pd.read_excel(data_file, sheet_name='Post Launch')
    irp_base = pd.read_excel(data_file, sheet_name='Base')
    x = pd.DataFrame()
    for c in irp_base['Country']:
        d = irp_base[irp_base['Country']==c]
        d['range'] = [list(range(d['Min '].values[0], d['Max'].values[0]+1))]
        x = x.append(d, ignore_index=True)
    irp_base = x.copy()
    fav_outcome = 1.1
    unfav_outcome = 0.9
    irp_base['Base Price'] = irp_base['Base Price']*fav_outcome*irp_base['Favourable Outcome'] + \
                            irp_base['Base Price']*unfav_outcome*irp_base['UnFavourable Outcome'] + \
                            irp_base['Base Price']*(1-(irp_base['Favourable Outcome']+irp_base['UnFavourable Outcome']))

    irp_at.rename(columns={'Primiary Basket':'Primary Basket', 'periodicity (In Months)':'periodicity'}, inplace=True)
    irp_post.rename(columns={'periodicity (In Moths)':'periodicity'}, inplace=True)
    irp_post['periodicity'].replace({'-':np.NaN, 'anually, can vary from year to year':12, '12\n':12}, inplace=True)
    irp_post['periodicity'].replace({'-':np.NaN}, inplace=True)
    irp_post['Multiplier'].replace({'Yes':1}, inplace=True)
    for d in [irp_at, irp_post]:
        d['Primary Basket'] = d['Primary Basket'].apply(lambda x: x.split(', ') if str(x)!='nan' else x)
        d['Primary Basket'] = d['Primary Basket'].apply(lambda x: [i.replace('\n', '').split(',') for i in x] if str(x)!='nan' else x)
        d['Primary Basket'] = d['Primary Basket'].apply(lambda x: [item for sublist in x for item in sublist] if str(x)!='nan' else x)
        d['Secondary Basket'] = d['Secondary Basket'].apply(lambda x: x.split(', ') if str(x)!='nan' else x)
        d['Pricing'] = np.where(d['Pricing'].str.contains('Free'), 'Free', d['Pricing'])
        d['Pricing'] = d['Pricing'].apply(lambda x: x.replace('\n', '') if '\n' in x else x)
        d['periodicity'] = np.where(d['periodicity']=='-', np.nan, d['periodicity'])
    irp_base['Launch Date'] = [str(i).split()[0] for i in irp_base['Launch Date']]
    df = price.drop(price.index[:])
    df['Country'] = irp_base['Country']
    d_list = []
    for c in df['Country'].unique():
        d = irp_base[irp_base['Country']==c]
        ds = df[df['Country']==c]
        ds[d['Launch Date'].values[0]] = d['Base Price']
        d_list.append(ds)
    df = pd.concat(d_list).fillna(0)
    x_df = pd.melt(df, id_vars=['Country'], value_vars=df.columns[1:].tolist()).sort_values(['Country', 'variable']).reset_index(drop=True)
    x_df.rename(columns={'variable':'Date', 'value':'Base Price'}, inplace=True)


    def main(seq, irp_data, start='2023-01-01'):
      global price, cogs, vol, dis, claw, pr, cgs_new, vold_df
      # '''
      #   seq: Sequence List of launch
      #   irp_data: irp base data
      #   start: first Launch date
      # '''
      start_date = start
      date_range = pd.date_range(start=start_date, periods=365*10, freq='D')
      date_range = [str(i).split()[0] for i in date_range.tolist() if '-01 ' in str(i)]
      seq_dic = dict(zip([ix+1 for ix, i in enumerate(date_range)], date_range))

      Countries = irp_data['Country'].tolist()
      cont_seq = dict(zip(Countries, seq))
      base_data = irp_data.copy()
      base_data['Launch Month'] = base_data['Country'].map(cont_seq)
      base_data['Launch Date'] = base_data['Launch Month'].map(seq_dic)
      base_data = base_data.dropna()
      df = price.drop(price.index[:])
      df['Country'] = base_data['Country']
      d_list = []
      for c in df['Country'].unique():
        d = base_data[base_data['Country']==c]
        ds = df[df['Country']==c]
        ds[d['Launch Date'].values[0]] = d['Base Price']
        d_list.append(ds)
      df = pd.concat(d_list).fillna(0)
      x_df = pd.melt(df, id_vars=['Country'], value_vars=df.columns[1:].tolist()).sort_values(['Country', 'variable']).reset_index(drop=True)
      x_df.rename(columns={'variable':'Date', 'value':'Base Price'}, inplace=True)

      irp_prices = IRP(base_long=x_df, base=base_data)
      # Recalculate Volume, Cogs, Disc, Claw
      cogs_df = cal_data(x_df, cogs, irp_prices)
      cgs_new = cogs_df.merge(cogs_seas, how='left')
      for col in [i for i in cogs_df.columns if i.startswith('20')]:
        if int(col.split('-')[1])<=3:
          cgs_new[col] = cgs_new[col]*cgs_new['Jan-Mar']
        elif int(col.split('-')[1])>3 and int(col.split('-')[1])<=6:
          cgs_new[col] = cgs_new[col]*cgs_new['April-Jun']
        elif int(col.split('-')[1])>6 and int(col.split('-')[1])<=9:
          cgs_new[col] = cgs_new[col]*cgs_new['Jul-Sep']
        else:
          cgs_new[col] = cgs_new[col]*cgs_new['Oct-Dec']
      cgs_new.drop(cgs_new.columns[-5:].tolist(), 1, inplace=True)

      vol_df = cal_data(x_df, vol, irp_prices)
      pr = pd.pivot_table(irp_prices, values=['Base Price'], index=['Country'], columns=['Date']).reset_index()
      pr.columns = ["_".join(tup).replace('Base Price_', '').replace('_', '') for tup in pr.columns.to_flat_index()]
      npv = NPV(pr, cgs_new, vol_df, dis, claw)
      # print(str(npv.iloc[:, 1:].sum(1).sum()/1e+06) + 'M')
      return npv, pr

    # Base case
    @st.cache(suppress_st_warning=True, show_spinner=False)
    def run_base(base):
      with st.spinner('Calculating Base case NPV'):
        s = base['Launch Month'].tolist() # sequence
        y_base, irp = main(s, base) # npv data
      return y_base, irp
    y, irp = run_base(irp_base)
    y = y.copy()
    npv = y.iloc[:, 1:].sum(1).sum()
    col1, col2 = st.columns(2)
    col1.header('Base Case Scenario')
    col1.write('Base case NPV: $ '+str(round(npv/1e+09, 4)) + 'B')
    col1.write(dict(irp_base[['Country', 'Launch Date']].values))
    my_expander = col1.expander("Show IRP updated prices")
    with my_expander:
      st.write(irp)

    st.sidebar.subheader('Constraints')
    N = st.sidebar.slider('Allowed number of launch countries in a month', 1, 8, 4)
    st.sidebar.write('Select Launch range of countries')
    ranges = []
    for cont, mi, ma in irp_base[['Country', 'Min ', 'Max']].values:
      values = st.sidebar.slider(cont, 1, 36, (mi, ma))
      ranges.append(values)
    ranges = [list(range(i[0], i[1]+1)) for i in ranges]
    opt_bt = st.sidebar.button(label='Optimize')
    st.sidebar.write('')

    if "opt_bt_state" not in st.session_state:
      st.session_state.opt_bt_state = False
        
    @st.cache(suppress_st_warning=True, show_spinner=False)
    def run_opt(base, launch_range, N):  
      final = base.copy()
      final['range'] = launch_range
      final['Launch Month'] = final['range'].apply(lambda x: x[0])
      with st.spinner('Optimizing for Best NPV...'):
        while final['Launch Month'].value_counts().reset_index()['Launch Month'].max()>N:
          seq_ = final['Launch Month'].tolist()
          # final['Launch Month'] = final['Launch']
          y_, _ = main(seq_, final)
          final = update_launch(y_, final, N)
          strt = dat(2023, 1, 1)
          new_dates = [str(strt + relativedelta(months=m-1)) for m in final['Launch Month']]
          final['Launch Date'] = new_dates
      s2 = final['Launch Month'].tolist()
      y2, irp_pr = main(s2, final)
      y2['best_npv'] = y2.iloc[:, 1:].sum(1)
      return final, y2, irp_pr

    def time_convert(sec):
      mins = sec // 60
      sec = sec % 60
      hours = mins // 60
      mins = mins % 60
      st.markdown(f"Time Taken - {int(sec)} secs")

    if opt_bt:
      start_time = time.time()
      st.session_state.opt_bt_state = True
      final, y2, irp_pr = run_opt(irp_base, ranges, N)
      y2 = y2.copy()
      end_time = time.time()
      time_lapsed = end_time - start_time
      time_convert(time_lapsed)

      col2.header('Optimized Scenario')
      col2.write('Best case NPV: $ '+str(round(y2['best_npv'].sum()/1e+09, 4)) + 'B')
      delta = round((y2['best_npv'].sum() - npv)/1e+06, 4)
      col2.write(dict(final[['Country', 'Launch Date']].values))
      my_expander = col2.expander("Show IRP updated prices")
      with my_expander:
          st.write(irp_pr)
      y['base_npv'] = y.iloc[:, 1:].sum(1)
      col1.write(y[['Country', 'base_npv']])
      col2.write(y2[['Country', 'best_npv']])
      # Delta visual
      vis_df = pd.merge(y[['Country', 'base_npv']], y2[['Country', 'best_npv']], on=['Country'])
      vis_df['diff'] = vis_df['best_npv'] - vis_df['base_npv']
      vis = vis_df.copy()
      vis_df = vis_df.sort_values(['best_npv'], ascending=False).set_index('Country')
      top_10 = round(vis_df["best_npv"][:10].sum()*100/vis_df["best_npv"].sum(), 2)
      bot_10 = round(vis_df["best_npv"][-10:].sum()*100/vis_df["best_npv"].sum(), 2)
      vis_df = vis_df[['diff']].reset_index(False).sort_values(['diff'], ascending=False)
      vis_df['cum_sum'] = vis_df['diff'].cumsum()
      vis_df['cum_perc'] = 100*vis_df['cum_sum']/vis_df['diff'].sum()
      vis_df['diff'] = vis_df['diff']/1e+06
      # st.write(vis_df)

      st.write(f'Delta: $ ', str(delta)+'M')
      st.write(f'Top 10 countries contribute to {top_10}% of NPV while bottom 10 are {bot_10}%')

      sns.set()
      fig, ax = plt.subplots(figsize=(18, 8))
      ax.bar(vis_df['Country'], vis_df["diff"], color="C0")
      ax2 = ax.twinx()
      ax2.plot(vis_df['Country'], vis_df["cum_perc"], color="C1", marker="D", ms=10)
      ax2.axhline(80, color="grey", linestyle="dashed")
      ax.set_ylabel('Delta ($ Millions)')
      ax.set_xticklabels(vis_df['Country'], rotation=90)
      ax2.yaxis.set_major_formatter(PercentFormatter())
      plt.title('Top contributors to delta in optimized sequence')
      st.pyplot(fig)

      lat_long = pd.read_csv('world_country_latitude_and_longitude.csv')
      lat_long.rename(columns={'country':'Country'}, inplace=True)
      vis = vis.merge(lat_long)
      def app():
          m = leafmap.Map(palette='RdBu_4')
          m.add_heatmap(
              vis,
              latitude="latitude",
              longitude="longitude",
              value="best_npv",
              name="Heat map",
              radius=20,
          )
          m.to_streamlit(height=500)
      app()

      @st.cache
      def convert_df(df):
          # IMPORTANT: Cache the conversion to prevent computation on every rerun
          return df.to_csv(index=False).encode('utf-8')
      if "download" not in st.session_state:
          st.session_state.download = False
      buffer_ = io.BytesIO()
      st.session_state.download = True
      with pd.ExcelWriter(buffer_, engine='openpyxl') as writer:
          y.to_excel(writer, sheet_name='Base Npv', index=False)
          irp.to_excel(writer, sheet_name='Base IRP prices', index=False)
          y2.to_excel(writer, sheet_name='Optimized Npv', index=False)
          irp_pr.to_excel(writer, sheet_name='Optimized IRP prices', index=False)
          writer.save()
          st.download_button(
                              label="Download Files",
                              data=buffer_,
                              file_name="Output.xlsx",
                              mime="application/vnd.ms-excel"
                          )
          st.info("*Files: Base NPV data | Base IRP Prices | Optimized NPV data | Optimized IRP Prices | Delta comparison", icon="ℹ️")
    else:
      st.stop()




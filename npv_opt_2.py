from collections import OrderedDict  # Import at the top of the file

def _calculate_irp(COUNTRY, launch_data, pre_data, n, base_df, date_launch, cont_df, is_at_launch=True):
    """Calculates IRP (International Reference Pricing) for a given country.

    Args:
        COUNTRY: Country name.
        launch_data: DataFrame containing launch-specific parameters (irp_at or irp_post).
        pre_data: DataFrame of IRP prices for previously launched countries.
        n: Number of lowest values to consider for 'Avg' calculation.
        base_df: DataFrame of base IRP data.
        date_launch: Launch date.
        cont_df: DataFrame of continuous date range.
        is_at_launch: Boolean flag indicating if it's at launch (True) or post-launch (False).

    Returns:
        A dictionary where keys are dates and values are IRP prices.
    """

    d = launch_data[launch_data['Country'] == COUNTRY].iloc[0]  # Access row once, avoid repeated lookups
    min_c = int(d['Min countries']) if isinstance(d['Min countries'], (int, float)) and not pd.isna(d['Min countries']) else 0 # Handle potential NaN or None values
    BASKET = d['Primary Basket'] or d['Secondary Basket']  # Use or operator for conciseness
    BASKET = list(filter(None, BASKET)) if isinstance(BASKET, str) and BASKET != 'nan' else [COUNTRY] # More robust nan check
    periodicity = int(d['periodicity']) if isinstance(d['periodicity'], (int, float)) and not pd.isna(d['periodicity']) else 12  # Default to 12 if nan or non-numeric
    at_period = 12 if periodicity <= 12 else periodicity
    
    end_date = cont_df[cont_df['Date'] > date_launch].reset_index(drop=True).iloc[:at_period]['Date'].iloc[-1]  # Use iloc for integer-based indexing
    date_range = cont_df[cont_df['Date'] <= end_date]['Date'].tolist()
    mul = float(d['Multiplier']) if isinstance(d['Multiplier'], (int, float)) and not pd.isna(d['Multiplier']) else 1.0  # Default to 1.0, handle nan or None
    irp_ref_month = int(d['IRP calculation month']) if isinstance(d['IRP calculation month'], (int, float)) and not pd.isna(d['IRP calculation month']) else 0

    price_dic = {}
    base_price = base_df[base_df['Country'] == COUNTRY]['Base Price'].iloc[0]  # Get base price once
    
    for date in date_range:
        if date < date_launch:
            price_dic[date] = 0
            continue  # Skip to the next iteration
        
        if not pd.isna(periodicity): # Check NaN using pandas function
            if date <= date_range[periodicity - 1]:
                price_dic[date] = base_price
            else:
                x_dt = date_range.index(date) - irp_ref_month if irp_ref_month else date_range.index(date)
                bask = [i for i in BASKET if not pre_data[(pre_data['Country'] == i) & (pre_data['Date'] == date_range[x_dt])].empty] # Use .empty for checking
                if len(bask) < min_c:
                    price_dic[date] = base_price
                else:
                    price_data = pre_data[(pre_data['Country'].isin(bask)) & (pre_data['Date'] == date_range[x_dt])]['Base Price']
                    if launch_data.name == "irp_at": # Differentiate between at_launch and post_launch
                        if d['at_launch'] == 'Min': # Access the at_launch value directly from the dataframe
                            price_dic[date] = price_data.min() * mul
                        elif d['at_launch'] == 'Avg':
                            price_dic[date] = price_data.sort_values().iloc[:n].mean() * mul
                            if COUNTRY == 'Denmark':
                                price_dic[date] = x_df[(x_df['Country'].isin(bask)) & (x_df['Base Price'] != 0)]['Base Price'].mean() * mul
                        elif d['at_launch'] == 'Free':
                            price_dic[date] = base_price * mul
                    else: # Post launch calculations
                        if d['post_launch'] == 'Min':
                            price_dic[date] = price_data.min() * mul
                        elif d['post_launch'] == 'Avg':
                            price_dic[date] = price_data.sort_values().iloc[:n].mean() * mul
                            # ... (Iceland, Denmark, Greece logic - refactor separately if possible)
                        elif d['post_launch'] == 'Free':
                            _d = list(dict(OrderedDict(sorted(at_dic.items(), reverse=True))).items())[0][1]
                            price_dic[date] = _d * mul if date == date_range[0] else price_dic[date_range[date_range.index(date)-1]]
            
        else: # Handle where periodicity is NaN
            _d = list(dict(OrderedDict(sorted(at_dic.items(), reverse=True))).items())[0][1]
            price_dic[date] = _d * mul if date == date_range[0] else price_dic[date_range[date_range.index(date)-1]]


    return price_dic



def IRP_AT(COUNTRY, at_launch, pre_data, n, base_df, date_launch, cont_df):
    global irp_at, x_df
    irp_at['at_launch'] = at_launch # Add at_launch to the irp_at dataframe
    return _calculate_irp(COUNTRY, irp_at, pre_data, n, base_df, date_launch, cont_df, is_at_launch=True)


def IRP_POST(COUNTRY, post_launch, at_dic, pre_data, n, cont_df):
    global irp_post
    irp_post['post_launch'] = post_launch # Add post_launch to the irp_post dataframe
    return _calculate_irp(COUNTRY, irp_post, pre_data, n, base_df, cont_df[cont_df['Date']>max(list(at_dic.keys()))]['Date'].tolist()[0], cont_df, is_at_launch=False)


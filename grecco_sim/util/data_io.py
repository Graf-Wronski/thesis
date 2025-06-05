import numpy as np
import pandas as pd


UNIX_TIME = "unixtimestamp"



def get_charging_data(ts_in, dt_h):
    # get parking duration ans soc data while ev is available for charging
    ts_out = pd.DataFrame(index=ts_in.index, columns=["until_departure", "initial_soc", "target_soc"])
    
    #soc in fraction
    ts_out["initial_soc"] = ts_in["soc_min_percent"]/100
    ts_out["target_soc"] = ts_in["soc_max_percent"]/100

    # ensure that soc data is available at each time step (for initialization)
    ts_out["initial_soc"].ffill(inplace = True)
    ts_out["target_soc"].bfill(inplace = True)

    # Filter non-zero values
    ts_ev_connected = ts_in["cp"][ts_in["cp"] != 0]

    # Create a grouping that increments if the time difference is not 15 minutes
    group = (ts_ev_connected.index.to_series().diff() != pd.Timedelta('15min')).cumsum()
    
    # Combine the series and group into a DataFrame
    df_temp = pd.concat([ts_ev_connected, group], axis=1)
    df_temp.columns = ["cp",'group']
    
    # Group by the block (using group and the value) and record start and end times
    availability = df_temp.groupby(['group']).apply(
        lambda x: pd.Series({
            'start_time': x.index[0],
            'end_time': x.index[-1]
        })
    ).reset_index(drop=True)

    #get time steps until departure (and interpolate soc)
    for idx, row in availability.iterrows():
                start_time = row["start_time"]
                end_time = row["end_time"]
                until_departure = (end_time - start_time)/dt_h
                ts_out.loc[start_time:end_time, "until_departure"] = np.arange(
                    until_departure, until_departure-len(ts_in.loc[start_time:end_time]),-1)               

    return ts_out

def set_tz_index_to_utc(df: pd.DataFrame) -> pd.DataFrame:
    """ df.tz_localize raises an Error on already localized data. This method
    implements it in a more robust fashion.

    Args:
        df: input dataframe.

    Returns:
        pd.DataFrame: The input dataframe with utc_localized index.
    """


    localized_df = df.copy()

    tz_index = pd.DatetimeIndex(df.index)

    if tz_index.tz is None:
        tz_index = tz_index.tz_localize("utc")
    else:
        tz_index = tz_index.tz_convert("utc")

    localized_df.index = tz_index

    return localized_df




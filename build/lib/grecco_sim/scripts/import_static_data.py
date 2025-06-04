# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:16:30 2024

@author: ekasper
"""

from pathlib import Path
import pandas as pd
import numpy as np

PATHS = {
    "loads": (Path(__file__).parent.parent.parent
              / "data" / "static_data_opfingen"
              / "loads.csv").resolve(),
    "generators": (Path(__file__).parent.parent.parent
                   / "data" / "static_data_opfingen"
                   / "generators.csv").resolve(),
    "storage_units": (Path(__file__).parent.parent.parent
                      / "data" / "static_data_opfingen"
                      / "storage_units.csv").resolve(),
    "synpro_ev_data": (Path(__file__).parent.parent.parent
                      / "data" / "static_data_opfingen"
                      / "synpro_ev_data_pool.csv").resolve(),
    "building_params": (Path(__file__).parent.parent.parent
                        / "data" / "static_data_opfingen" /
                        "building_params.csv").resolve(),
    }


# Function to import the CSV files
def import_csv():
    data = {}
    for key, path in PATHS.items():
        try:
            if key == "building_params":
                data[key] = pd.read_csv(path, sep=';')
            elif key == "amsportplatz":
                data[key] = pd.read_excel(path, sheet_name="load", engine='openpyxl')
            else:
                data[key] = pd.read_csv(path)
            print(f"Successfully loaded: {key}")
        except FileNotFoundError:
            print(f"File not found: {path}")
        except pd.errors.ParserError as e:
            print(f"Error loading file {path} - ParserError: {e}")
        except UnicodeDecodeError as e:
            print(f"Encoding error with file {path}: {e}")
        except Exception as e:
            print(f"Error loading file {path}: {e}")
    # Delete the last two rows of generators (Testgen and Slack)
    data["generators"] = data["generators"].iloc[:-2]
    data["loads"]["carrier"].fillna("household", inplace=True)
    
    return data


# Function to populate the DataFrame
def create_dataframe(data):
    df = pd.DataFrame(columns=['element_index', 'bus','name','pv_p_mw', 
                               'bat_p_mw', 'bat_c_mhw',
                               'ev_p_mw', 'ev_c_kwh',
                               'ev_eff_dis', 'ev_eff_ch',
                               'hp_p_mw', 'hp_cop', 'hp_ref',
                               'building_u', 'building_cp',
                               'building_abs', 'building_irr_area'])

    # PV generators
    if 'generators' in data:
        gen_df = data['generators'][['carrier','bus', 'p_set']].copy()
        gen_df.rename(columns={'carrier': 'name', 'p_set': 'pv_p_mw'}, inplace=True)
        gen_df['element_index'] = gen_df.index  
        # Assuring same format in index
        df = pd.concat([df, gen_df], ignore_index=True)

    # Storage Units
    if 'storage_units' in data:
        storage_df = data['storage_units'][['name', 'bus', 'p_nom']].copy()
        storage_df['element_index'] = data['storage_units'].index  
        
        # Separate into ev_df and battery_df
        ev_df = storage_df[storage_df['name'].str.contains('emob')].copy()
        ev_df.rename(columns={'p_nom': 'ev_p_mw'}, inplace=True)
        ev_df['element_index'] = list(range(len(data['loads']), len(data['loads'])+len(ev_df)))
        battery_df = storage_df[storage_df['name'].str.contains('battery')].copy()
        battery_df.rename(columns={'p_nom': 'bat_p_mw'}, inplace=True)
        battery_df['element_index'] = list(range(0,len(battery_df)))
        # Filling battery capacity as 2 times its p_nom
        battery_df['bat_c_mhw'] = battery_df['bat_p_mw'] * 2  # TODO This is a workaround
        
        synpro_ev_df = data['synpro_ev_data'][['name',
                                                'efficiency_dispatch',
                                                'efficiency_store',
                                                'capacity_kWh']].copy()

        synpro_ev_df.set_index('name', inplace=True)
        # Initialize new columns
        ev_df['ev_c_kwh'] = None
        ev_df['ev_eff_dis'] = None
        ev_df['ev_eff_ch'] = None
        # Populate ev_df with values from synpro_ev_df
        for name, bus in zip(ev_df['name'], ev_df.index):
            try:
                ev_df.loc[bus, 'ev_c_kwh'] = synpro_ev_df.loc[name, 'capacity_kWh']
                ev_df.loc[bus, 'ev_eff_dis'] = synpro_ev_df.loc[name, 'efficiency_dispatch']
                ev_df.loc[bus, 'ev_eff_ch'] = synpro_ev_df.loc[name, 'efficiency_store']
            except KeyError:
                print(f"EV ID in node {bus} not found in synpro pool")
                print(f"For EV in node {bus}, default parameters are assigned")

                # Assign mean values as default
                ev_df.loc[bus, 'ev_c_kwh'] = synpro_ev_df['capacity_kWh'].mean()
                ev_df.loc[bus, 'ev_eff_dis'] = synpro_ev_df['efficiency_dispatch'].mean()
                ev_df.loc[bus, 'ev_eff_ch'] = synpro_ev_df['efficiency_store'].mean()

        df = pd.concat([df, battery_df], ignore_index=True)
        df = pd.concat([df, ev_df], ignore_index=True)

    # Heat Pumps
    if 'loads' in data:
        load_df = data['loads'][['carrier', 'bus', 'p_set']].copy()
        load_df.rename(columns={'carrier': 'name'}, inplace=True)
        heatpump_df = load_df[load_df['name'].str.contains('heat')].copy()
        heatpump_df.rename(columns={'p_set': 'hp_p_mw'}, inplace=True)
        heatpump_df['element_index'] = list(range(0,len(heatpump_df)))

        df = pd.concat([df, heatpump_df], ignore_index=True)

    # Adding building thermal parameters
    df = populate_building_parameters(df, data)
    

    return df


def populate_building_parameters(df, data):
    building_params_df = data["building_params"]
    building_types = list(building_params_df['class'])
    loads_list = data["loads"][['bus', 'name', 'p_set', 'carrier']].copy()
    loads_list = loads_list[loads_list['carrier'] == 'household'] 
    loads_list.loc[loads_list['p_set'] == 0.0025, 'name'] = 'EFH'
    loads_list.loc[loads_list['p_set'] == 0.0035, 'name'] = 'ZFH'
    loads_list.loc[loads_list['p_set'] >= 0.0045, 'name'] = 'MFH'
    building_list = loads_list[loads_list['name'].isin(building_types)]
    building_list['element_index'] = building_list.index
    building_list.index = building_list.index.astype(str)
    building_df = building_list.copy() 
    
    # Clean and structure building_params DataFrame
    building_params_df.columns = ['class', 'original_class', 'building_u', 'building_cp', 'building_abs', 'building_irr_area']
    building_params_df[['building_u', 'building_cp', 'building_abs', 'building_irr_area']] = \
        building_params_df[['building_u', 'building_cp', 'building_abs', 'building_irr_area']].astype(float)
    building_params_df.set_index('class', inplace=True)
    building_params_df.index = building_params_df.index.astype(str)
    building_params_df.drop(columns=['original_class'], inplace=True)

    # Copying thermal parameters to the building_df
    for parameter in building_params_df.columns:
        building_df[parameter] = None
        for name, bus in zip(building_df['name'], building_df.index):
            building_df.loc[bus, parameter] = building_params_df.loc[name, parameter]

    building_df.drop(columns=['carrier'], inplace=True)

    df = pd.concat([df, building_df], ignore_index=True)

    return df


# Save the DataFrame to CSV
def save_dataframe_to_csv(df, filepath):
    try:
        df.to_csv(filepath, index=False)
        print(f"DataFrame successfully saved at: {filepath}")
    except Exception as e:
        print(f"Error saving file: {e}")


def main():
    # Import CSV files
    csv_data = import_csv()
    # Create and populate the main DataFrame
    df = create_dataframe(csv_data)
    # Save the populated DataFrame
    save_path = Path(__file__).parent.parent.parent / "data" / "static_data_opfingen" / "static_data.csv"
    save_dataframe_to_csv(df, save_path)


if __name__ == "__main__":
    main()

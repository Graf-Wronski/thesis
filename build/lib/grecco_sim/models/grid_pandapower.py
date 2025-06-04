from pathlib import Path
import pandas as pd
import pandapower as pp 
from pandapower.timeseries import DFData
import pandapower.control as control
import pandapower.timeseries as timeseries
from pandapower.timeseries.run_time_series import run_timeseries
import numpy as np
import pytz


#%% VARIABLES

START = "2023-08-01 00:00"
END = "2023-08-01 23:45"

CALC_REACTIVE_Q = True

COSPHI = {
    'load': 0.98,
    'sgen': 1,
    'storage': 1}

OUTPUT_VAR = [['res_bus','p_mw'],
              ['res_bus','q_mvar'],
              ['res_bus','vm_pu'],
              #['res_bus','va_degree'],
              ['res_load','p_mw'],
              ['res_load','q_mvar'],
              ['res_sgen', 'p_mw'],
              ['res_sgen', 'q_mvar'],
              ['res_line', 'i_ka'],
              ['res_line', 'loading_percent'],
              ['res_trafo', 'loading_percent'],
              ['res_trafo', 'p_hv_mw'],
              #['res_trafo', 'p_lv_mw'],
              ['res_trafo', 'q_hv_mvar'],
              #['res_trafo', 'q_lv_mvar'],
              #['res_ext_grid', 'p_mw'],
              #['res_ext_grid', 'q_mvar'],
              ['res_storage', 'p_mw'],
              ['res_storage', 'q_mvar']]

#%% GRID CLASS

class Grid(object):
    def __init__(self):
        # Initialize the Grid object and create a power grid for opfingen
        self.create_grid('opfingen')
        
        
    def create_grid(self, path):
        """
        Loads a pandapower grid based on the specified path.
        
        Args:
            path (str): Identifier for the grid configuration 
            (e.g., 'opfingen').
        
        Returns:
            net (pandapowerNet): The loaded pandapower grid object.
        """
        if path == 'opfingen':
            # Load the grid model from a JSON file
            self.net = pp.from_json((Path.cwd().parent.parent
                    / "data" / "grid_model"
                    / "AmSportplatz_Opfingen_lv_gis052024_coords_real.json"))
            self._init_grid_elements()
        #else: 
            #TO DO if necessary
        
        print("Grid object created with pandapower.")
        net = self.net
        
        return net
        
    
    def _init_grid_elements(self):
        """
        Initialize the grid elements by removing existing static generators,
        storage units and loads and creating new ones from a CSV file.
        """
        self.delete_sgen_storage_loads()
        self.create_sgen_storage_loads_from_csv()
        
        return


    def get_grid_elements(self):
        """
        Return grid elements from the pandapower network.
        
        Returns:
            DataFrames of different grid elements 
            (e.g. loads, static generators).
        """
        load = self.net.load
        sgen = self.net.sgen
        storage = self.net.storage
        bus = self.net.bus
        line = self.net.line
        trafo = self.net.trafo
        ext_grid = self.net.ext_grid
        
        return load, sgen, storage, bus, line, trafo, ext_grid
    

    def get_trafo_laoding(self):
        """
        Load transformer loading results from Excel files.
        
        Returns:
            dict: Contains loading data for transformers 
        """
        trafo_variables = ['loading_percent', 'p_hv_mw', 'q_hv_mvar']
        trafo_res = {}
        
        for var in trafo_variables:
            file_path = (Path.cwd().parent.parent / "data"  / "grid_model" / 
                         "powerflow_results" / "res_trafo" / f"{var}.xlsx")
            if file_path.exists():
                try:
                    # Read transformer results from Excel file
                    res = pd.read_excel(file_path, index_col=0).drop(columns=[0], 
                                        errors='ignore')
                except Exception as e:
                    print(f"Error reading {var}: {e}")
            trafo_res[var] = res
        
        return trafo_res
    
    
    def delete_sgen_storage_loads(self):
        """
        Deletes all static generators (sgen), storage units
        and loads from the pandapower network.
        """
        if not self.net.sgen.empty:
            self.net.sgen.drop(self.net.sgen.index, inplace=True)
        
        if not self.net.storage.empty:
            self.net.storage.drop(self.net.storage.index, inplace=True)

        if not self.net.load.empty:
            self.net.load.drop(self.net.load.index, inplace=True)

        if not self.net.ward.empty:
            self.net.ward.drop(self.net.ward.index, inplace=True)    
        
        print("All sgen, storage and loads have been deleted from the grid.")
        
        return


    def create_sgen_storage_loads_from_csv(self, path = (Path.cwd().parent.parent
            / "data" / "static_data_opfingen"
            / "static_data.csv")):
        """
        Creates static generators, storage units and loads from a CSV file.
        
        Args:
            path (pathlib.Path): Path to the CSV file containing static data.
        """
        # Load data from the CSV file
        data = pd.read_csv(path)
        
      
        for _, row in data.iterrows():
            bus = row['bus']
            element_index = row['element_index']
            name = row['name']
            
            # Add elements based on the data in the CSV file
            if not pd.isna(row['pv_p_mw']):
                pp.create_sgen(self.net, 
                               name=name, 
                               bus=bus, 
                               p_mw=row['pv_p_mw'], 
                               scaling = 1,
                               index = element_index)
            
            
            elif not pd.isna(row['hp_p_mw']):
                pp.create_load(self.net, 
                               name=name, 
                               bus=bus, 
                               p_mw=row['hp_p_mw'], 
                               scaling = 1,
                               index = element_index)
                
            elif not pd.isna(row['building_u']):
                pp.create_load(self.net, 
                               name=name, 
                               bus=bus, 
                               p_mw=row['p_set'], 
                               scaling = 1,
                               index = element_index)
                
            elif not pd.isna(row['ev_p_mw']):
                pp.create_load(self.net, 
                               name=name, 
                               bus=bus, 
                               p_mw=row['ev_p_mw'], 
                               scaling = 1,
                               index = element_index)
                
            elif not pd.isna(row['bat_p_mw']):
                pp.create_storage(self.net, 
                                  name=name, 
                                  bus=bus, 
                                  max_e_mwh=row['bat_c_mhw'], 
                                  p_mw=row['bat_p_mw'], 
                                  scaling = 1,
                                  index = element_index)
                
        print("New sgen, storage and loads have been added from the CSV file.")
        
        return
        
    
    def prepare_timeseries_synpro_for_power_flow(self, data):
        """
        Prepares time series for the controllers.
        COLUMN NAME MUST BE THE SAME VALUE AS THE ELEMENT INDEX NOT THE BUS
        COLUMN NAME MUST BE INTEGER

        Args:
            data (dict): Dictionary containing time series data.
        """
        
        loads_buses = pd.read_csv((Path.cwd().parent.parent
                  / "data" / "Opfingen_Profiles_2023" 
                  / "Opfingen_scenario_pv_20_ev_efh_0_ev_mfh_0_evghd_0_ev_fleet_0_hp_15_2023_with_h0_batterypf_all"
                  / "loads.csv"),
            usecols=['name', 'bus', 'carrier'])
        # IMPORTANT: Change column names to element index
        # Mapping from 'name' to index in loads_buses
        name_to_index = {str(row['name']): str(idx) for idx, row in loads_buses.iterrows()}
        # New column names based on the mapping
        new_columns = {col: name_to_index[col] for col in data['loads'].columns if col in name_to_index}
        # Rename columns in loads_p_set
        data['loads'].rename(columns=new_columns, inplace=True)
        
        
        # Iterate through column names in 'emob'
        for column in data['emob'].columns:
            # Match column names with loads in the Pandapower network
            if column in self.net.load.name.values:
                # Find the corresponding load index
                load_index = self.net.load.index[self.net.load.name == column][0]
                # Rename the column in 'data['emob']' to match the index
                data['emob'].rename(columns={column: load_index}, inplace=True)

        # Iterate through column names in 'storage_units'
        for column in data['storage_units'].columns:
            # Match column names with storage elements in the Pandapower network
            if column in self.net.storage.name.values:
                # Find the corresponding storage index
                storage_index = self.net.storage.index[self.net.storage.name == column][0]
                # Rename the column in 'data['storage_units']' to match the index
                data['storage_units'].rename(columns={column: storage_index}, inplace=True)
        
        # Set heat pump loads to zero for specific time ranges
        self.set_heat_pump_loads_to_zero(data)
        
        for key in data.keys():
            if isinstance(data[key], pd.DataFrame):
                # Convert column names to integers for Pandapower compatibility
                data[key].columns = data[key].columns.astype(int)  
        
        # Combine 'emob' and 'loads' DataFrames along the columns
        data['loads'] = pd.concat([data['loads'], data['emob']], axis=1)
        # Remove 'emob' key from the dictionary as it's now merged into 'loads'
        del data['emob']
        print("Timeseries for power flow have been prepared.")
        
        return data


    def set_heat_pump_loads_to_zero(self, data):
        """
        Sets all 'heat_pump'  values to zero for a specified time range 
        (May to October).

        Args:
            data (dict): Dictionary containing time series data.
        """
        # Identify heat pump loads by name
        for idx in self.net.load.index:
            load_name = self.net.load.name[idx]
            if 'heat_pump' in load_name:
                # Ensure the column exists in 'loads'
                if str(idx) in data['loads'].columns:
                    # Set the specified time range to zero
                    data['loads'].loc[11520:26208, str(idx)] = 0
        return data
    

    def input_var_for_timeseries_sim(self, data):
        """
        Creates a dictionary defining input variables for time series simulation.

        Args:
            data (dict): Dictionary containing time series data.

        Returns:
            dict: A dictionary mapping variable names to their simulation parameters.
        """
        input_var = {
            'loads_p': ['load', 'p_mw', data['loads'], 1.0],
            'loads_q': ['load', 'q_mvar', None, 1.0],
            'generators_p': ['sgen', 'p_mw', data['generators'], 1.2],
            'generators_q': ['sgen', 'q_mvar', None, 1.0],
            'storage_units_p': ['storage', 'p_mw', data['storage_units'], 1.0],
            'storage_units_q': ['storage', 'q_mvar', None, 1.0]
        }
        
        return input_var
    
    
    def create_control_dataframes(self, input_var):
        """
        Creates control DataFrames based on the defined input variables.

        Args:
            input_var (dict): elements and their simulation parameters

        Returns:
            dict: dictionary containing control DataFrames
        """
        ctr_df = {}
        for key, value in input_var.items():
            # Check if it's a valid DataFrame
            if isinstance(input_var[f"{key}"][2], pd.DataFrame) and not input_var[f"{key}"][2].empty:
                ctr_df[f"{key}"] = input_var[f"{key}"][2] 
            else:
                ctr_df[f"{key}"] = None  # Otherwise, set to None

        return ctr_df
    
    
    def assign_reactive_power(self, ctr_df, input_var, COSPHI):
        """
        Assigns reactive power values if not already provided, 
        based on active power and cosphi values.

        Args:
            ctr_df (dict): control DataFrames
            input_var (dict): elements and their simulation parameters
            COSPHI (dict): Dictionary containing cosphi values for each element.

        Returns:
            Updated control DataFrames and input variables.
        """
        # Iterate over non-None items in ctr_df
        for key in {k: v for k, v in ctr_df.items() if v is not None}:  
            # Check if the key is related to active power
            if '_p' in key and ctr_df[key] is not None:  
                # Construct the corresponding reactive power key
                reactive_key = key[:-2] + '_q'  
                
                # Check if reactive power is predefined
                if reactive_key not in ctr_df or ctr_df[reactive_key] is None:  
                    # Define the cos_phi value depending on load, sgen or storage
                    var_name = f"{input_var[key][0]}"
                    # Get the corresponding cosphi value from COSPHI dictionary
                    var_value = COSPHI.get(var_name)  
                    
                    if var_value is not None:  # Ensure that a valid cosphi value is found
                        # Calculate reactive power using q = p * tan(arccos(cosphi))
                        ctr_df[reactive_key] = np.tan(np.arccos(var_value)) * ctr_df[key]
                        # Update the input_var to reflect the newly generated reactive power data
                        input_var[reactive_key] = [input_var[key][0], 
                                                   'q_mvar', 
                                                   'automatically generated', 
                                                   input_var[key][3]]
                    else:
                        print(f"Warning: Cosphi value for '{var_name}' not found in COSPHI dictionary.")
                        
        return ctr_df, input_var
    
    
    def create_controllers(self, ctr_df, input_var):
        """
        Creates controllers for each defined control variable (e.g., active and reactive power).

        Args:
            ctr_df (dict): control DataFrames
            input_var (dict): elements and their simulation parameters
        """
        # Lists to collect DataFrames based on element types
        load_p_dfs_to_concat = []
        load_q_dfs_to_concat = []
        sgen_p_dfs_to_concat = []
        sgen_q_dfs_to_concat = []

        # Iterate over the control DataFrames and categorize based on the element type
        for key, df in ctr_df.items():
            if df is not None and key.endswith('_p') and input_var[key][0]=='load':
                load_p_dfs_to_concat.append(df) # Add active power DataFrames for loads
                load_p_scaling = input_var[key][3]
            elif df is not None and key.endswith('_q') and input_var[key][0]=='load':
                load_q_dfs_to_concat.append(df) # Add reactive power DataFrames for loads
                load_q_scaling = input_var[key][3]
            elif df is not None and key.endswith('_p') and input_var[key][0]=='sgen':
                sgen_p_dfs_to_concat.append(df) # Add active power DataFrames for generators
                sgen_p_scaling = input_var[key][3]
            elif df is not None and key.endswith('_q') and input_var[key][0]=='sgen':
                sgen_q_dfs_to_concat.append(df) # Add reactive power DataFrames for generators
                sgen_q_scaling = input_var[key][3]
            elif df is not None and key.endswith('_p') and input_var[key][0]=='storage':
                df_storage_p = df.copy() # Copy for storage units
                storage_p_scaling = input_var[key][3]
            elif df is not None and key.endswith('_q') and input_var[key][0]=='storage':
                df_storage_q = df.copy() # Copy for storage units' reactive power
                storage_q_scaling = input_var[key][3]
                # Convert the time series to a controller for storage
                self.convert_timeseries_to_controller(df_storage_p, 'storage', 'p_mw', storage_p_scaling)
                self.convert_timeseries_to_controller(df_storage_q, 'storage', 'q_mvar', storage_q_scaling)
                
        # Combine the DataFrames for active and reactive power based on load and generator types
        combined_df_loads_p = pd.concat(load_p_dfs_to_concat, axis=1)
        sorted_columns = [i for i in self.net.load.index if i in combined_df_loads_p.columns]
        combined_df_loads_p = combined_df_loads_p[sorted_columns]
        
        combined_df_loads_q = pd.concat(load_q_dfs_to_concat, axis=1)
        sorted_columns = [i for i in self.net.load.index if i in combined_df_loads_q.columns]
        combined_df_loads_q = combined_df_loads_q[sorted_columns]
        
        # Convert the combined DataFrames to controllers for load
        self.convert_timeseries_to_controller(combined_df_loads_p, 'load', 'p_mw', load_p_scaling)
        self.convert_timeseries_to_controller(combined_df_loads_q, 'load', 'q_mvar', load_q_scaling)
        
        

        
        if sgen_p_dfs_to_concat != []:
            # Combine the DataFrames for active and reactive power for generators
            combined_df_sgen_p = pd.concat(sgen_p_dfs_to_concat, axis=1)
            sorted_columns = [i for i in self.net.sgen.index if i in combined_df_sgen_p.columns]
            combined_df_sgen_p = combined_df_sgen_p[sorted_columns]
            
            combined_df_sgen_q = pd.concat(sgen_q_dfs_to_concat, axis=1)
            sorted_columns = [i for i in self.net.sgen.index if i in combined_df_sgen_q.columns]
            combined_df_sgen_q = combined_df_sgen_q[sorted_columns]
            
            # Convert the combined DataFrames to controllers for generators    
            self.convert_timeseries_to_controller(combined_df_sgen_p, 'sgen', 'p_mw', sgen_p_scaling)
            self.convert_timeseries_to_controller(combined_df_sgen_q, 'sgen', 'q_mvar', sgen_q_scaling)

        print("Controller for each defined control variable are created.")
        
        return
    

    def convert_timeseries_to_controller(self, df, cont_element, variable, scale):
        """
        Converts a DataFrame of time series data to a pandapower controller.
        
        Parameters:
        - df: DataFrame with time series data.
        - cont_element: The element to control ('load', 'sgen', etc.).
        - variable: The variable to control ('p_mw', 'q_mvar', etc.).
        - scale: Scale factor for the data (Standard is defined as 1.0, only changes if defined otherwise in main file (time_series_simulation.py))
        """

        n_ts = df.shape[0]
        df.columns = self.net[cont_element].index
        df.index = list(range(n_ts))
        # Convert DataFrame to data source
        ds = DFData(df)
        control.ConstControl(self.net, element=cont_element, element_index=self.net[cont_element].index,
                     variable=variable, data_source=ds, profile_name=self.net[cont_element].index, scale_factor = scale)
        return

    
    def create_output_writer(self, output_path = (Path.cwd().parent.parent
            / "data" / "grid_model"
            / "powerflow_results"), output_var = OUTPUT_VAR):
        """
        Function to initialize the output writer
        
        Args:
            output_path (Path): Path to the directory where the output will be saved.
            output_var (list): List of variables to log in the output file.

        Returns:
            OutputWriter: An instance of the output writer.
        """
        ow = timeseries.OutputWriter(self.net, output_path=output_path, output_file_type=".xlsx")
        for var in output_var:
            table, variable = var
            ow.log_variable(str(table), str(variable))  # Log each variable to the output writer
            
        print("Output writer created.")
        return ow
    
    
    def safe_timestamp(self, start, end):
        """
        Saves the time period for plotting by converting the start and end 
        times to a DataFrame.

        Args:
            start (str): Start time as a string.
            end (str): End time as a string.

        """
        # Convert start and end to datetime objects
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        
        # Create a time series at a 15-minute frequency
        time_series = pd.date_range(start=start, end=end, freq="15T")

        # Create a DataFrame with the timestamps
        df = pd.DataFrame({"Timestamp": time_series})

        output_path = (Path.cwd().parent.parent
                / "data" / "grid_model"
                / "powerflow_results")
        
        output_file = output_path / "time_series.xlsx"

        # Create the output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)

        # Save the DataFrame as an Excel file
        df.to_excel(output_file, index=False)
        
        return 
    
    
    
    def prepare_control_data(self, data, input_var, OUTPUT_VAR, COSPHI, 
                             calc_reactive_q=True):
        """
        Main function to prepare control data for time series simulation.
        
        Args:
            data (dict): Dictionary containing time series data.
            input_var (dict): Dictionary of input variables for control.
            OUTPUT_VAR (list): List of output variables to log.
            COSPHI (dict): Dictionary mapping element types to cos(phi) values.
            calc_reactive_q (bool): Flag to determine whether to calculate 
            reactive power (default: True).
        
        Returns:
            ctr_df (dict): Control dataframes for each variable.
        """
        # Create control dataframes based on input variables
        ctr_df = self.create_control_dataframes(input_var)  
        # Assign reactive power values
        ctr_df, input_var = self.assign_reactive_power(ctr_df, input_var, COSPHI)  
        
        return ctr_df

    
    def start_grid_sim(self, data):
        """
        Main function to start the grid simulation by preparing data, 
        creating controllers, initializing output writing and running the simulation.
        
        Args:
            data (dict): Dictionary containing time series data.
        """
        # Prepare the time series data for the power flow simulation
        data = self.prepare_timeseries_synpro_for_power_flow(data)
        # Define input variables for the simulation
        input_var = self.input_var_for_timeseries_sim(data)
        # Prepare control dataframes for the simulation
        ctr_df = self.prepare_control_data(data, input_var, OUTPUT_VAR, COSPHI)
        # Create controllers for each defined control variable
        self.create_controllers(ctr_df, input_var)  # Create controllers
        self.create_output_writer()  # Create the output writer
        # Save timestamps for simulation visualization
        self.safe_timestamp(START, END)
        print("Run power flow simulation:")
        # Execute the time series simulation
        run_timeseries(self.net)
        
        return
    
    

#%% TIMESERIES DATA CLASS

class TimeSeriesData(object):
    
    def import_timeseries_synpro(self, start, end, schedules_agends = None): 
        """
        Loads time series data from CSV files and prepares it for use.
        
        Args:
            start (str): Start time in a datetime format (string).
            end (str): End time in a datetime format (string).
            schedules_agends (optional): TO DO
        
        Returns:
            dict: A dictionary containing the filtered time series data for different elements.
        """
        # Define Berlin timezone
        berlin_tz = pytz.timezone("Europe/Berlin")
        
        # Convert start and end times to Berlin time zone
        start = pd.to_datetime(start).tz_localize("UTC").tz_convert(berlin_tz)
        end = pd.to_datetime(end).tz_localize("UTC").tz_convert(berlin_tz)
        
        # Define the start of the year and frequency (15-minute interval)
        start_of_year = pd.Timestamp("2023-01-01 00:00:00", tz="UTC").tz_convert(berlin_tz)
        freq = pd.Timedelta(minutes=15)  
        
        # Define file paths for time series data
        paths_timeseries = {
            "loads": (Path.cwd().parent.parent
                      / "data" / "Opfingen_Profiles_2023" 
                      / "Opfingen_scenario_pv_20_ev_efh_0_ev_mfh_0_evghd_0_ev_fleet_0_hp_15_2023_with_h0_batterypf_all"
                      / "loads-p_set.csv"),
            "generators": (Path.cwd().parent.parent
                      / "data" / "Opfingen_Profiles_2023"
                      / "Opfingen_scenario_pv_20_ev_efh_0_ev_mfh_0_evghd_0_ev_fleet_0_hp_15_2023_with_h0_batterypf_all"
                      / "generators-p_set.csv"),
            "storage_units": (Path.cwd().parent.parent
                      / "data" / "Opfingen_Profiles_2023" 
                      / "Opfingen_scenario_pv_20_ev_efh_0_ev_mfh_0_evghd_0_ev_fleet_0_hp_15_2023_with_h0_batterypf_all"
                      / "storage_units-p_set.csv")
        }
        
        # Helper function to convert timestamps to indices
        def timestamp_to_index(timestamp):
            delta = (timestamp - start_of_year) // freq
            return int(delta)
        
        # Calculate start and end indices based on the time range
        start_index = timestamp_to_index(start)
        end_index = timestamp_to_index(end)
        
        
        data = {}
        for key, path in paths_timeseries.items():
            try:
                # Read the CSV file into a DataFrame
                df = pd.read_csv(path)     
                # Cut the data range to the requested time period
                if key != 'generators':  # Filter out the 'generators' key as it does not need filtering
                    filtered_data = df.iloc[start_index:end_index + 1]  
                    data[key] = filtered_data
                    print(f"Successfully loaded: {key}")
                else: 
                    data[key] = df # For generators, time range filtering must be after scaling
                    print(f"Successfully loaded: {key}")
            except FileNotFoundError:
                print(f"File not found: {path}")
            except Exception as e:
                print(f"Error loading the file {path}: {e}")
                
        # Split the 'storage_units' DataFrame into two parts: one for 'emob' and one for 'battery'
        data['emob'] = data['storage_units'].loc[:, data['storage_units'].columns.str.contains('emob')] 
        data['emob'] = data['emob'] * -1 # Invert the values for pandapower
        # Filter the 'storage_units' DataFrame to only include columns related to 'battery'
        data['storage_units'] = data['storage_units'].loc[:, data['storage_units'].columns.str.contains('battery')] 
        data['storage_units'] = data['storage_units'] * -1 # Invert the values for pandapower
        
        # Drop any "Unnamed" columns that may exist in the 'generators' data
        data['generators'] = data['generators'].drop(columns=['Unnamed: 0'], errors='ignore')
        # Scale the 'loads' data 
        data['loads'] = data['loads'].mul(3, axis=1)
        
        # Scale the PV profiles (photovoltaic power generation profiles)
        self.scale_generator_profiles(data)
        # Now filter the 'generators' data to the selected time range
        data['generators'] = data['generators'].iloc[start_index:end_index + 1]
        
        return data      
        
    
    def scale_generator_profiles(self, data):
        """
        Scales the PV profiles in the 'generators' data based on predefined capacities.
        
        Args:
            data (dict): Dictionary containing the time series data for various elements.
        
        Modifies:
            data['generators']: Scales the generator profiles according to the capacities.
        """
        # Predefined list of PV capacities in MWp (megawatt peak)
        inst_power_pv_mw = [
            0.00507, 0.0068, 0.0054, 0.00792, 0.00966, 0.00992, 0.00558,
            0.00855, 0.0076, 0.0108, 0.00195, 0.00273, 0.00594, 0.01632,
            0.01212, 0.0055, 0.01536, 0.01496, 0.00496, 0.00252, 0.00252,
            0.00252, 0.00252, 0.00496, 0.00324, 0.0033, 0.00574, 0.0072,
            0.0006, 0.01260, 0.00984, 0.00840, 0.00560, 0.00600]
        
        # Copy the 'generators' data for scaling
        gen_p_set_for_scaling = data['generators'].copy()
        for col in gen_p_set_for_scaling.columns:
            gen_p_set_for_scaling[col] = gen_p_set_for_scaling['0'] # Set all columns to the first column's values
            
        # Calculate the reference annual yield for the PV profile in kWh (kilowatt-hours)
        # Convert MW to kW and divide by 4
        jahresertrag_referenzprofil_pv_kWh = gen_p_set_for_scaling['0'].sum() * 1000 / 4   
        spezif_jahresertrag_opfingen = 1148  # kWh/kWp (specific annual yield in kWh per installed kWp)
        # Calculate the annual yield for each PV system in kWh
        jahresertrag_pv_kwh = [wert * (1000 *spezif_jahresertrag_opfingen) for wert in inst_power_pv_mw]
        # Calculate the scaling factor for each PV system
        scaling_factor_pv = [(wert / jahresertrag_referenzprofil_pv_kWh) for wert in jahresertrag_pv_kwh]
            
        # Apply the scaling to the generator profiles
        data['generators'] = gen_p_set_for_scaling.mul(scaling_factor_pv, axis=1)
            
        return data


#%% MAIN

# def main():
    # pp_net = Grid()
    # timeseries_data = TimeSeriesData()

    # data = timeseries_data.import_timeseries_synpro(START, END)
    # pp_net.start_grid_sim(data)
    # trafo_res = pp_net.get_trafo_laoding()

    
# if __name__ == "__main__":
#     main()

pp_net = Grid()
timeseries_data = TimeSeriesData()

data = timeseries_data.import_timeseries_synpro(START, END)
pp_net.start_grid_sim(data)
trafo_res = pp_net.get_trafo_laoding()

#load, sgen, storage, bus, line, trafo, ext_grid = pp_net.get_grid_elements()


#%% Plots ... 

#from pandapower.plotting.plotly import pf_res_plotly
#pf_res_plotly(pp_net.net)

# import matplotlib.pyplot as plt
# loads = ctr_df['loads_p']
# columns_to_plot = loads[[i for i in range(0, 20)]]
# # Plotten
# columns_to_plot.plot(figsize=(10, 6))
# plt.title("heatpumps")
# plt.xlabel("time")
# plt.ylabel("p_mw")
# plt.legend(title="index_heatpumps")
# plt.show()

# remaining_columns = [col for col in loads.columns if col not in columns_to_plot]
# columns_to_plot_loads = loads[remaining_columns]
# # Plotten
# columns_to_plot_loads.plot(figsize=(10, 6))
# plt.title("household loads")
# plt.xlabel("time")
# plt.ylabel("p_mw")
# plt.legend(title="index_loads")
# plt.show()
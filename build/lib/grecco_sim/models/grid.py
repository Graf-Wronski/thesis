import pathlib
import warnings
from grecco_sim.util import logger
from grecco_sim.models import model_base
import pandas as pd
import numpy as np
import pypsa
import pathlib

class Grid(object):
    def __init__(self, path: pathlib.Path):
        self.network = pypsa.Network()

        print("Reading pypsa files ...")
        with warnings.catch_warnings(action="ignore"):
            self.network.import_from_csv_folder(path)
        print("... Done.")

        self._init_variables()
        print("Grid object created")

    @staticmethod
    def _sys_id(load: int, bus: int):
        return f"bus_{bus}_load_{load}"

    def _init_variables(self):
        """
        Decision:
            Loads are aggregated on a bus level.
            Later distinction between several loads at each bus is possible.
        """

        _grid: pypsa.Network = self.network

        _loads = _grid.loads.loc[_grid.loads.carrier != "heat_pump", :]
        self._ind_load = _loads.index.to_list()

        _hps =  _grid.loads.loc[_grid.loads.carrier == "heat_pump", :]
        self._ind_hps = _hps.index.to_list()

        _gens = _grid.generators[_grid.generators.carrier == "solar"]
        self._ind_gens = _gens.index.to_list()

        # Take unique buses
        self._buses = list(set(_loads.bus))

        # Assert generators and storages are connected to a bus where also a load is connected
        assert set(_loads.bus) == set(_loads.bus) | set(_hps.bus) | set(_gens.bus) | set(_grid.storage_units.bus)

        self.sys_ids = [
            self._sys_id(l, b)
            for b in self._buses
            for l in _loads.loc[_loads.bus == b, :].index
        ]

        self.load_p = self.network.loads_t["p_set"].loc[:, self._ind_load]
        self.load_p = self.load_p.rename(
            columns={
                i: self._sys_id(i, self.network.loads.loc[i, "bus"])
                for i in self._ind_load
            }
        )
        self.load_p *= 1000.  # Transform to kW instead of MW

        # Generate << a >> bus to load mapping
        bus_to_load = _loads.reset_index().groupby("bus").first()["Load"]

        # rename and scale PV input data
        self.pv_p = self.network.generators_t["p_set"].loc[:, self._ind_gens]
        self.pv_p = self.pv_p.rename(
            columns={
                i: self._sys_id(bus_to_load.loc[_gens.bus[i]], _gens.bus[i]) for i in _gens.index
            }
        )
        self.pv_p *= 1000.  # Transform to kW instead of MW

        self.generators = _gens.rename(index={
            i: self._sys_id(bus_to_load.loc[_gens.bus[i]], _gens.bus[i]) for i in _gens.index
        })
        self.generators["p_set"] *= 1000.
        self.storages = _grid.storage_units

        # Rename and scale heat pumps
        self.heat_pumps = _hps.rename(index={
            i: self._sys_id(bus_to_load.loc[_hps.bus[i]], _hps.bus[i]) for i in _hps.index
        })
        # ToDo: Not the place for data adjustments.
        self.heat_pumps["p_nom"] = self.heat_pumps["pLoad"]
        self.heat_pumps["p_nom"] *= 1000.

    def get_load_ts(self) -> pd.DataFrame:
        return self.load_p
    
    def get_pv_ts(self) -> pd.DataFrame:
        return self.pv_p

    def get_pv_data(self):
        installed_pv = self.network.generators[self.network.generators.carrier == 'solar'].p_set
        pv_generation = self.network.generators_t.p_set.loc[:, installed_pv.index]
        # To calculate the capacity factor as time series we have to divide the generation over installed_pv
        capacity_factor = pv_generation / installed_pv
        return {'installed_pv':installed_pv, 'pv_generation':pv_generation, 'capacity_factor':capacity_factor}

    def get_ev_data(self):
        ev_capacity = self.network.storage_units.p_nom
        plugged_in = self.network.storage_units_t.plugged_in.loc[:, ev_capacity.index]
        soc_departure_max_percent = self.network.storage_units_t.soc_departure_max_percent
        soc_departure_min_percent = self.network.storage_units_t.soc_departure_min_percent

        return {'ev_capacity':ev_capacity,
                 'plugged_in':plugged_in, 
                 'soc_departure_max_percent':soc_departure_max_percent,
                 'soc_departure_min_percent':soc_departure_min_percent}

    def get_load_data(self):
        load_index = self.network.loads[self.network.loads.carrier != 'heat_pump'].index
        load_buses = self.network.loads.loc[load_index, 'bus']
        load_active_power = self.network.loads_t.p_set.loc[:, load_index]
        try:
            load_reactive_power = self.network.loads_t.q_set.loc[:, load_index]
        except KeyError:
            # Since no reactive power data is available, we assume that the reactive power is 0
            # This has to be changed in the future to include realistic reactive power data otherwise 
            # the power flow calculation might not converge
            # within iterations limit, assuming all buses are PV buses would be incorrect since this 
            # won't allow us to see voltage drops (Voltage control assumption)
            load_reactive_power = load_active_power*0

        return {'load_index':load_index,
                'load_bus_map':load_buses,
                'load_active_power':load_active_power,
                'load_reactive_power': load_reactive_power}

    def get_heatpump_data(self):
        heatpump_index = self.network.loads[self.network.loads.carrier == 'heat_pump'].index
        heatpump_buses = self.network.loads.loc[heatpump_index, 'bus']
        heatpump_rated_power = self.network.loads.loc[heatpump_index, ['p_set', 'q_set']]
        heatpump_active_power_profile = self.network.loads_t.p_set.loc[:, heatpump_index]
        return { 'heatpump_index':heatpump_index,
                'heatpump_buses':heatpump_buses,
                'heatpump_rated_power':heatpump_rated_power,
                'heatpump_active_power_profile':heatpump_active_power_profile}

    def calculate_timeseries_powerflow(self, n_timesteps: int = 672, path = None, export_csv = False): 
        """
        Function calculates power flow and generates csv files (time series data). 

        Required to check congestion, visualize the bus voltage levels, and avoid 
        repetitive power flow calculations (which can be time-consuming).
        """     
        # Non-linear power flow
        self.network.pf(self.network.snapshots[:n_timesteps]) 

        # Export the results 
        if export_csv: 
            self.network.export_to_csv_folder(path)
        
        return self.network

    def _set_active_load(self, load_set:pd.Series):
        '''
        This methods is used to set the power of each load in the grid.
        Only the loads that are in the load_set will be set. Other loads will remain unchanged.
        This allows the user to set Heatpump and EV loads separately.

        :param load_set: DataFrame with the columns as load name and the index as the snapshot range

        return: None
        '''

    def _set_reactive_load(self, load_set:pd.Series):
        '''
        This methods is used to set the reactive power of each load in the grid.
        Only the loads that are in the load_set will be set. Other loads will remain unchanged.
        This allows the user to set Heatpump and EV loads separately.

        :param load_set: DataFrame with the columns 'q_mvar' and the index as the load id

        return: None
        '''
        load_set.columns = ['q_mvar']

        self.net.load.q_mvar.loc[load_set.index, :] = load_set

    def _set_active_sgen(self, sgen_set:pd.Series):
        '''
        This methods is used to set the power of each sgen in the grid.
        Only the sgens that are in the sgen_set will be set. Other sgens will remain unchanged.
        This allows the user to set PV generation separately.

        :param sgen_set: DataFrame with the columns 'p_mw' and the index as the sgen id

        return: None
        '''
        sgen_set.columns = ['p_mw']

        self.net.sgen.p_mw.loc[sgen_set.index, :] = sgen_set

    def _set_reactive_sgen(self, sgen_set:pd.Series):
        '''
        This methods is used to set the reactive power of each sgen in the grid.
        Only the sgens that are in the sgen_set will be set. Other sgens will remain unchanged.
        This allows the user to set PV generation separately.

        :param sgen_set: DataFrame with the columns 'q_mvar' and the index as the sgen id

        return: None
        '''
        sgen_set.columns = ['q_mvar']

        self.net.sgen.q_mvar.loc[sgen_set.index, :] = sgen_set

    def check_congestion(self):
        """
        Function for evaluating the state of the grid regarding the congestion.

        Return: 
            congestion_results: dict
                Dictionary containing the KPIs of the congestion calculations.
        """

        # Check if time series data is available (e.g. for lines)
        count = 0
        for key, df in self.network.lines_t.items():
            if df.empty:
                count += 1

        if count == len(self.network.lines_t):
            raise Exception("Calculate the power flow using the 'calculate_timeseries_powerflow' function, \n"
                            "or import the results of the power flow calculations if they are available.")
        else:
        
            # Evaluating transformer loading 
            # Trafo_loading= (Power_transfered(at time t)/ Nominal capacity)*100
            trafo_nominal_capacity = self.network.transformers["s_nom"]
            trafo_power_transferred = np.sqrt(
                self.network.transformers_t["p1"] ** 2 + self.network.transformers_t["q1"] ** 2)
            #trafo_power_transferred = self.network.transformers_t["p1"]
            trafo_loading_perc = (trafo_power_transferred / trafo_nominal_capacity) * 100

            # Events of congestion in the transformer 
            # Events_of_congestion= 1 if Trafo_loading>80% and 0 if Trafo_loading<=80%
            trafo_events_of_congestion = (trafo_loading_perc > 80).astype(int)
            trafo_total_events_of_congestion = trafo_events_of_congestion.sum() 

            # Evaluating line loading
            line_nominal_capacity = self.network.lines["s_nom"]
            line_power_transferred = np.sqrt(self.network.lines_t["p1"] ** 2 + self.network.lines_t["q1"] ** 2)
            #line_power_transferred = self.network.lines_t["p1"]
            line_loading_perc = (line_power_transferred / line_nominal_capacity) * 100

            # CLI = CL / G (congestion line index)
            line_events_of_congestion = (line_loading_perc > 80).astype(int)
            line_total_events_of_congestion = line_events_of_congestion.sum()
            count_congested_lines = (line_total_events_of_congestion > 0).sum()
            congestion_line_index = count_congested_lines / len(self.network.lines)

            # CVI = sum of power transferred / installed line capacity 
            # (congestion volume index) - this considers only one timestep 
            congestion_volume_index = (line_power_transferred.iloc[0] / line_nominal_capacity).sum()

            # Weighted average is performed to consider the importance of line capacity; 
            # large lines are more important than the smaller ones
            # Weighted average = weight * max_power_flow in line / line capacity
            # weight = line capacity / sum of all line capacities 
            max_power_flow_line = line_power_transferred.max() 
            weight = line_nominal_capacity / line_nominal_capacity.sum()
            weighted_average = 0 
            for i in range(len(max_power_flow_line)):
                weighted_average += weight[i] * max_power_flow_line[i] / line_nominal_capacity[i]
            
            congestion_results = {
            "trafo_loading_perc": trafo_loading_perc,
            "trafo_total_events_of_congestion": trafo_total_events_of_congestion,
            "line_loading_perc": line_loading_perc,
            "line_total_events_of_congestion": line_total_events_of_congestion,
            "congestion_line_index": congestion_line_index,
            "congestion_volume_index": congestion_volume_index,
            "weighted_average": weighted_average
            } 

            return congestion_results
             

    def estimate_power_flow(self):
        # return ptdf matrix for the network
        # pp.rundcpp(self.net)
        # _, ppci = _pd2ppc(self.net)

        # ptdf_sparse = makePTDF(ppci["baseMVA"], ppci["bus"], ppci["branch"],
        #                     using_sparse_solver=True)
        # return ptdf_sparse
        pass

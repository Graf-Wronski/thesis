import pathlib
import pandas as pd
import matplotlib.pyplot as plt

class ResultsToPypsa():

    # Initialization
    def __init__(self, directory_optimization_results: pathlib.Path, 
                 directory_pypsa_results: pathlib.Path,
                 file_optimization_results: str, file_component_bus_connection: str, file_to_be_updated: str):
        
        self.directory_optimization_results = directory_optimization_results
        self.directory_pypsa_results = directory_pypsa_results
        self.file_optimization_results = file_optimization_results
        self.file_component_bus_connection = file_component_bus_connection
        self.file_to_be_updated = file_to_be_updated

        # Import optimization results 
        self.optimization_results = pd.read_csv(
            self.directory_optimization_results / f"{self.file_optimization_results}.csv", sep = ";")
        self.optimization_results = self.optimization_results.drop(
            self.optimization_results.columns[-1], axis = 1) # Drop "unixtimestamp" column 
        self.optimization_results.set_index(self.optimization_results.columns[0], inplace = True)

        # Import general info (component connection to which bus)
        self.component_bus_connection = pd.read_csv(
            self.directory_pypsa_results / f"{self.file_component_bus_connection}.csv", sep = ",")

        # Import active power as time series 
        self.component_p_series = pd.read_csv(self.directory_pypsa_results / f"{self.file_to_be_updated}.csv",
                                              sep = ",")
        self.component_p_series.set_index(self.component_p_series.columns[0], inplace = True)
        self.component_p_series = self.component_p_series.iloc[:self.optimization_results.shape[0], :]
        self.optimization_results.index = self.component_p_series.index # Matching the indices 

    # Update the PyPSA network 
    def transfer_results_to_pypsa(self, export_csv = True):

        # Copy the dataframe 
        self.component_p_series_updated = self.component_p_series.copy()

        # Get the bus number connected to each component  
        bus_numbers = []
        for i in self.component_p_series_updated.columns:
            bus_numbers.append(
                self.component_bus_connection.loc[self.component_bus_connection["name"] == int(i), "bus"].values)
        bus_numbers = pd.Series(bus_numbers)
 
        # Replace the values 
        for col in self.optimization_results.columns:
            if "bat" in col:
                bus_part_number = int(col.split("_")[1]) # Extracting the bus number 
                load_number_part = int(col.split("_")[3]) # Extracting the load number 
                if bus_part_number in bus_numbers.values:
                    matching_index = bus_numbers[bus_numbers == bus_part_number].index[0]
                    self.component_p_series_updated.iloc[:, matching_index] = self.optimization_results[
                        f"bus_{bus_part_number}_load_{load_number_part}_bat_p_ac"].values.reshape(-1, 1) 
            
            if "hp" in col: 
                bus_part_number = int(col.split("_")[1]) # Extracting the bus number 
                load_number_part = int(col.split("_")[3]) # Extracting the load number 
                if bus_part_number in bus_numbers.values:
                    matching_index = bus_numbers[bus_numbers == bus_part_number].index[0]
                    self.component_p_series_updated.iloc[:, matching_index] = self.optimization_results[
                        f"bus_{bus_part_number}_load_{load_number_part}_hp_p_ac"].values.reshape(-1, 1) 
            
            
            if "ev" in col:
                bus_part_number = int(col.split("_")[1]) # Extracting the bus number 
                load_number_part = int(col.split("_")[3]) # Extracting the load number 
                if bus_part_number in bus_numbers.values:
                    matching_index = bus_numbers[bus_numbers == bus_part_number].index[0]
                    self.component_p_series_updated.iloc[:, matching_index] = self.optimization_results[
                        f"bus_{bus_part_number}_load_{load_number_part}_ev_p_ac"].values.reshape(-1, 1) 
        
        # Create the csv file
        if export_csv:
            path = pathlib.Path(__file__).parent.absolute()
            path = path.parent.parent
            path = path / "results" / f"{self.file_to_be_updated}_updated.csv"
            self.component_p_series_updated.to_csv(path, index = True)

        return self.component_p_series_updated
    
    def plot_component_time_series(self):
        plt.plot(self.component_p_series)
        plt.xlabel("Timesteps")
        plt.ylabel("MW")
        plt.title("Active Power Over Time - Original")
        plt.show()

        plt.plot(self.component_p_series_updated)
        plt.xlabel("Timesteps")
        plt.ylabel("MW")
        plt.title("Active Power Over Time - Updated")
        plt.show()
    
if __name__ == "__main__":
    
    # Instantiate the class
    class_instance = ResultsToPypsa(directory_optimization_results = 
                                    pathlib.Path(__file__).parent.parent.parent.absolute() / 
                                    "results" / "parallelized" / "test_availables" / "second_order_it_5",
                                    directory_pypsa_results = 
                                    pathlib.Path(__file__).parent.parent.parent.absolute() / 
                                    "results" / "power_flow_results",
                                    file_optimization_results = "flex_power_second_order_it_5",
                                    file_component_bus_connection = "storage_units",
                                    file_to_be_updated = "storage_units-p_set"
                                    )
    
    # Update component time series 
    component_p_series_updated = class_instance.transfer_results_to_pypsa(export_csv = True)

    # Plot component time series before and after coordination 
    class_instance.plot_component_time_series()
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:40:02 2024

@author: ekasper
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from IPython import get_ipython
import matplotlib.cm as cm
import numpy as np
import pandapower as pp 

get_ipython().run_line_magic('matplotlib', 'qt')

def import_res(file_path, start_datetime, end_datetime):
    """
    Lädt Daten aus einer Excel-Datei und fügt Zeitstempel hinzu.
    """
    data = pd.read_excel(file_path, index_col=0).drop(columns=[0], errors='ignore')
    num_rows = len(data)
    time_index = pd.date_range(start=start_datetime, end=end_datetime, periods=num_rows)
    data.index = time_index
    return data

def plot_results(data, title, x_label, y_label, parameter, category, net):
    """
    Erstellt und zeigt einen Plot der übergebenen Daten.
    """
    
    if category == 'res_sgen':
        legend_labels = []
        for col in data.columns:
            # Suche den Wert aus sgen, der dem Spaltennamen entspricht
            bus_value = net.sgen.loc[col, 'bus']  # sgen.loc[col] gibt dir die Zeile für den Index 'col'
            legend_labels.append(f'pv_node_{bus_value}')  # Formatieren der Legende
    elif category == 'res_load':
        legend_labels = []
    elif category == 'res_storage':
        legend_labels = []
        for col in data.columns:
            # Suche den Wert aus sgen, der dem Spaltennamen entspricht
            bus_value = net.storage.loc[col, 'bus']  # sgen.loc[col] gibt dir die Zeile für den Index 'col'
            legend_labels.append(f'storage_node_{bus_value}')  # Formatieren der Legende        
    elif category == 'res_trafo':
        legend_labels = []
    elif category == 'res_line':
        legend_labels = []
        # for col in data.columns:
        #     legend_labels.append(f'line_index_{col}')  # Formatieren der Legende      
    elif category == 'res_bus':
        legend_labels = []
        # for col in data.columns:
        #     legend_labels.append(f'bus_index_{col}')  # Formatieren der Legende         
            
            
    fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
    
    if parameter == 'vm_pu' or category == 'res_line':  # Scatter-Plot für vm_pu
        # Generiere eine Farbpalette mit so vielen Farben wie Spalten in den Daten
        num_columns = len(data.columns)
        colors = cm.tab20(np.linspace(0, 1, num_columns))  # Verwende z. B. die "tab20"-Palette
    
        # Iteriere über die Spalten und plotte jede mit eigener Farbe
        for i, column in enumerate(data.columns):
           ax.scatter(data.index, data[column], s=2, c=[colors[i]], alpha=1, label=column, marker='d')
        # Berechne den Zeitraum basierend auf dem Index des DataFrames
        time_span = data.index[-1] - data.index[0]  # Differenz zwischen dem letzten und dem ersten Timestamp

        # Bestimme den internen Locator basierend auf der Zeitspanne
        if time_span < pd.Timedelta(days=1):  # Wenn der Zeitraum weniger als ein Tag ist
            locator = mdates.HourLocator(interval=12)  # 12 Stunden-Intervall
        elif time_span < pd.Timedelta(days=30):  # Wenn der Zeitraum weniger als ein Monat ist
            locator = mdates.DayLocator(interval=1)  # 6 Stunden-Intervall
        else:  # Wenn der Zeitraum mehr als ein Monat beträgt
            locator = mdates.MonthLocator(interval=1)  # 1 Tag-Intervall
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b \n %H:%M'))  # Zeigt Datum und Uhrzeit an
        ax.xaxis.set_major_locator(locator)
        ax.grid(axis='x')
        
    else:  # Linienplot für andere Parameter
        num_columns = len(data.columns)
        colors = cm.tab20(np.linspace(0, 1, num_columns))
        data.plot(ax=ax, legend=None, color=[colors[i] for i in range(num_columns)])
    
    if category == 'res_load':
        ax.legend(legend_labels)
        ax.legend(legend_labels, ncol=3, fontsize=2, loc='upper left', bbox_to_anchor=(1, 1))
    else:
        ax.legend(legend_labels)
        ax.legend(legend_labels, ncol=2, fontsize=3, loc='best')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid()
    plt.tight_layout()
    
    
        
    
def main():
    # Pfade zu den Dateien
    base_path = Path.cwd().parent.parent / "data" / "grid_model" / "powerflow_results"
    
    net = pp.from_json((Path.cwd().parent.parent
            / "data" / "grid_model"
            / "AmSportplatz_Opfingen_lv_gis052024_coords_real.json"))
    
    timestamp_file = base_path / "time_series.xlsx"
    
    # Zeitstempel laden
    timestamps = pd.read_excel(timestamp_file)
    start_datetime = timestamps.iloc[0, 0]
    end_datetime = timestamps.iloc[-1, 0]

    # Variablen zum Plotten
    OUTPUT_VAR = [
        ['res_bus','p_mw'],
        #['res_bus','q_mvar'],
        ['res_bus','vm_pu'],
        #['res_bus','va_degree'],
        ['res_load','p_mw'],
        #['res_load','q_mvar'],
        ['res_sgen', 'p_mw'],
        #['res_sgen', 'q_mvar'],
        #['res_line', 'i_ka'],
        ['res_line', 'loading_percent'],
        ['res_trafo', 'loading_percent'],
        ['res_trafo', 'p_hv_mw'],
        #['res_trafo', 'p_lv_mw'],
        #['res_trafo', 'q_hv_mvar'],
        #['res_trafo', 'q_lv_mvar'],
        #['res_ext_grid', 'p_mw'],
        #['res_ext_grid', 'q_mvar'],
        ['res_storage', 'p_mw'],
        #['res_storage', 'q_mvar']
    ]

    # Variablen durchlaufen und plotten
    for var in OUTPUT_VAR:
        category, parameter = var
        file_path = base_path / category / f"{parameter}.xlsx"
        
        if file_path.exists():
            try:
                data = import_res(file_path, start_datetime, end_datetime)
                if parameter == 'vm_pu':
                    data = data.drop(columns=[320, 321, 324, 325])
                elif parameter == 'p_mw' and category == 'res_bus':
                    data = data.drop(columns=[325])
                elif parameter == 'q_mvar' and category == 'res_bus':
                    data = data.drop(columns=[325])
                    
                plot_results(data, 
                             title=f"{category} - {parameter} results",
                             x_label="Time",
                             y_label=f"{parameter}",
                             parameter=parameter,
                             category=category,
                             net=net)
            except Exception as e:
                print(f"Fehler beim Plotten von {category}/{parameter}: {e}")
        else:
            print(f"Datei für {category}/{parameter} nicht gefunden: {file_path}")

if __name__ == "__main__":
    main()
    

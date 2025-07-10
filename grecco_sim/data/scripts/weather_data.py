from pathlib import Path
import pandas as pd


def main():
    data_root = Path("/home/carl-wanninger/data")
    input_path = data_root / "weather" / "test" / "pvgis_2016_00_download.csv"
    output_path = data_root / "weather" / "test" / "pvgis_2016_00.csv"

    # Read PVGIS input.
    csv_input = pd.read_csv(input_path, header=6)

    # Adjust columns.
    # csv_input = csv_input.drop(["H_sun", "WS10m", "Int"], axis=1)

    # Expand data from hourly to quarterhourly.
    timestamps = pd.Series(pd.date_range(
        start="2016-01-01",
        periods=4*len(csv_input),
        freq="15min"),
    )
    data = {"Outside Temperature": csv_input["T2m"].repeat(4).to_list(),
            "Solar Irradiance": csv_input["G(i)"].repeat(4).to_list()}
    pd.DataFrame(index=timestamps, data=data).to_csv(output_path)

    # Save csv.

if __name__ == "__main__":
    main()

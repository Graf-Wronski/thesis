import datetime
import pathlib
from matplotlib import pyplot as plt
import pandas as pd



def read_fc(sys_id: str) -> pd.DataFrame:
    data_path_root = pathlib.Path("/home/agross/data/grecco/forecasts/dimitrios_v1/Outputs")

    df = pd.read_csv(data_path_root / f"{sys_id}.csv", infer_datetime_format=True, index_col=0, parse_dates=True)
    return df


def get_statistics(sys_id: str):


    fc = read_fc(sys_id)
    dt = fc.index[1] - fc.index[0]
    
    fc = fc.iloc[:20, :]

    

    fc.plot()

    plt.show()


if __name__ == "__main__":
    get_statistics("76")




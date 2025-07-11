import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

from grecco_sim.graph.utils.format import Format


def get_represntative_days(year: int):
    p = Format().data_root / "weather" / f"{year}_dwd.csv"
    df = pd.read_csv(p)
    # 1. Load your data
    # Assuming df has columns: ['datetime', "Outside Temperature", "Solar Irradiance"]
    df['datetime'] = pd.to_datetime(df['Date'])
    df = df.set_index('datetime')

    # 2. Reshape into daily profiles
    daily_profiles = df.groupby(df.index.date).apply(
        lambda x: np.concatenate(
            [x["Outside Temperature"].values, x["Solar Irradiance"].values])
            if not x.isnull().any().any() and len(x) == 96 else None
    ).dropna()

    daily_profiles = pd.DataFrame(daily_profiles.tolist(),
                                  index=pd.to_datetime(daily_profiles.index))

    # 3. Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(daily_profiles)

    # 4. Apply KMeans
    k = 4  # Number of representative days you want
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(scaled_data)

    # 5. Find representative day of each cluster
    representative_days = []
    for cluster_id in range(k):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_points = scaled_data[cluster_indices]
        centroid = kmeans.cluster_centers_[cluster_id]

        # Compute distances to centroid
        distances = cdist(cluster_points, [centroid])
        closest_index = cluster_indices[np.argmin(distances)]

        representative_days.append(daily_profiles.index[closest_index])

    return representative_days

if __name__ == "__main__":
    for year in [2016, 2023]:
        representative_days = get_represntative_days(year)

        # Output
        print(f"Representative Days for year {year}:")
        for day in representative_days:
            print(day.date())

import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from hysoc.core.point import Point
from hysoc.core.segment import Stop, Move
from benchmarks.oracles.stss_sklearn import STSSOracleSklearn

def load_trajectory(filepath):
    df = pd.read_csv(filepath)
    points = []
    for _, row in df.iterrows():
        # Adjust column names based on actual CSV format. 
        # Assuming standard names or I'll need to check the file content if this fails.
        # Based on previous context, user has 'latitude', 'longitude', 'timestamp'
        
        # Check standard variations
        lat_col = 'latitude' if 'latitude' in df.columns else 'lat'
        lon_col = 'longitude' if 'longitude' in df.columns else 'lon'
        time_col = 'timestamp' if 'timestamp' in df.columns else 'time'
        id_col = 'id' if 'id' in df.columns else 'oid'
        
        try:
            ts = pd.to_datetime(row[time_col])
        except:
            ts = datetime.now() # Fallback, shouldn't happen
            
        points.append(Point(
            lat=float(row[lat_col]),
            lon=float(row[lon_col]),
            timestamp=ts,
            obj_id=str(row.get(id_col, 'unknown'))
        ))
    return points

def tune_parameters(filepath):
    print(f"Loading {filepath}...")
    try:
        traj = load_trajectory(filepath)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Loaded {len(traj)} points.")

    # Parameter grid
    min_samples_list = [2, 3, 5]
    max_eps_list = [20.0, 50.0, 100.0, 200.0]
    min_duration_list = [10.0, 30.0, 60.0, 120.0]

    results = []

    print(f"{'MinSam':<8} {'MaxEps':<8} {'MinDur':<8} | {'Stops':<6} {'Moves':<6} | {'AvgStopDur':<10}")
    print("-" * 65)

    for min_samples in min_samples_list:
        for max_eps in max_eps_list:
            for min_duration in min_duration_list:
                oracle = STSSOracleSklearn(
                    min_samples=min_samples, 
                    max_eps=max_eps, 
                    min_duration_seconds=min_duration
                )
                segments = oracle.process(traj)
                
                n_stops = sum(1 for s in segments if isinstance(s, Stop))
                n_moves = sum(1 for s in segments if isinstance(s, Move))
                
                stop_durations = [
                    (s.end_time - s.start_time).total_seconds() 
                    for s in segments if isinstance(s, Stop)
                ]
                avg_stop_dur = sum(stop_durations) / len(stop_durations) if stop_durations else 0

                res_str = f"{min_samples:<8} {max_eps:<8} {min_duration:<8} | {n_stops:<6} {n_moves:<6} | {avg_stop_dur:<10.2f}"
                print(res_str)
                with open("scripts/tuning_results.txt", "a") as f:
                    f.write(res_str + "\n")
                results.append((min_samples, max_eps, min_duration, n_stops))

    best_result = max(results, key=lambda x: x[3])
    with open("scripts/tuning_results.txt", "a") as f:
        f.write("\nConfiguration with most stops:\n")
        f.write(f"Min Samples: {best_result[0]}, Max Eps: {best_result[1]}, Min Duration: {best_result[2]} -> {best_result[3]} stops\n")
    print("\nConfiguration with most stops:") 
    print(f"Min Samples: {best_result[0]}, Max Eps: {best_result[1]}, Min Duration: {best_result[2]} -> {best_result[3]} stops")

if __name__ == "__main__":
    target_file = project_root / "data/raw/subset_50/4494499.csv"
    tune_parameters(target_file)

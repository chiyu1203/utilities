# this is a file to convert data from matrexVR to locustVR.
# Input: csv file, gz csv file from matrexVR
# output: h5 file that stores single animal's response in multiple conditions
import time
import pandas as pd
import numpy as np
import os, gzip, re, csv, json, sys
from pathlib import Path
from funcs import *
from threading import Lock
from useful_tools import find_file, find_nearest

"""
demo if you want to load packages from other directories
current_working_directory = Path.cwd()
parent_dir = current_working_directory.resolve().parents[0]
sys.path.insert(0, str(parent_dir) + "\\utilities")

"""
lock = Lock()


def ffill(arr):
    mask = np.isnan(arr)
    if arr.ndim == 1:
        print("work in progress")
        # idx = np.where(~mask, np.arange(mask.shape[0]), 0)
        # np.maximum.accumulate(idx, out=idx)
        # out = arr[np.arange(idx.shape[0])[None], idx]
    elif arr.ndim == 2:
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        out = arr[np.arange(idx.shape[0])[:, None], idx]
    return out


# Simple solution for bfill provided by financial_physician in comment below
def bfill(arr):
    if arr.ndim == 1:
        return ffill(arr[::-1])[::-1]
    elif arr.ndim == 2:
        return ffill(arr[:, ::-1])[:, ::-1]


def read_simulated_data(this_file, analysis_methods):
    print("read simulated data")
    if type(this_file)==str:
        this_file=Path(this_file)
    if this_file.suffix == ".gz":
        with gzip.open(this_file, "rb") as f:
            df = pd.read_csv(f)
    elif this_file.suffix == ".csv":
        with open(this_file, mode="r") as f:
            df = pd.read_csv(f)
    print(df.columns)
    n_locusts = df.columns[6]
    boundary_size = df.columns[7]
    mu = df.columns[8]
    kappa = df.columns[9]
    simulated_speed = df.columns[10]
    density = int(n_locusts.split(":")[1]) / (
        int(boundary_size.split(":")[1]) ** 2 / 10000
    )
    conditions = {
        "Density": density,
        mu.split(":")[0]: int(mu.split(":")[1]),
        kappa.split(":")[0]: float(kappa.split(":")[1]),
        simulated_speed.split(":")[0]: float(simulated_speed.split(":")[1]),
    }
    if len(df) > 0:
        ts = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d %H:%M:%S.%f")
        x = df["X"]
        y = df["Z"]
    else:
        ts = pd.to_datetime(this_file.stem[0:19], format="%Y-%m-%d_%H-%M-%S")
        x = None
        y = None
    return ts, x, y, conditions


def align_matrex_data(
    this_file,
    analysis_methods,
    ts_simulated_animal,
    x_simulated_animal,
    y_simulated_animal,
    conditions,
):
    print("read locust data")
    # track_ball_radius = analysis_methods.get("trackball_radius")
    # monitor_fps = analysis_methods.get("monitor_fps")
    # camera_fps = analysis_methods.get("camera_fps")
    BODY_LENGTH = analysis_methods.get("body_length")
    growth_condition = analysis_methods.get("growth_condition")
    generate_locust_vr_matrices = analysis_methods.get("generate_locust_vr_matrices")
    overwrite_curated_dataset = analysis_methods.get("overwrite_curated_dataset")
    heading_direction_across_trials = []
    x_across_trials = []
    y_across_trials = []
    ts_across_trials = []
    if type(this_file)==str:
        this_file=Path(this_file)
    if this_file.suffix == ".gz":
        with gzip.open(this_file, "rb") as f:
            df = pd.read_csv(f)
    elif this_file.suffix == ".csv":
        with open(this_file, mode="r") as f:
            df = pd.read_csv(f)
    print(df.columns)
    df["SensPosX"].replace(0.0, np.nan, inplace=True)
    df["SensPosY"].replace(0.0, np.nan, inplace=True)

    df["Current Time"] = pd.to_datetime(
        df["Current Time"], format="%Y-%m-%d %H:%M:%S.%f"
    )
    experiment_id = df["VR"][0] + " " + str(df["Current Time"][0]).split(".")[0]
    experiment_id = re.sub(r"\s+", "_", experiment_id)
    experiment_id = re.sub(r":", "", experiment_id)
    curated_file_path = this_file.parent / f"{experiment_id}_XY.h5"
    summary_file_path = this_file.parent / f"{experiment_id}_score.h5"
    if overwrite_curated_dataset == True and curated_file_path.is_file():
        curated_file_path.unlink()
        try:
            summary_file_path.unlink()
        except OSError as e:
            # If it fails, inform the user.
            print("Error: %s - %s." % (e.filename, e.strerror))

    df.set_index(df["Current Time"], inplace=True)
    for id in range(len(ts_simulated_animal)):
        this_ts = ts_simulated_animal[id]
        try:
            df[df.index == this_ts]
            if (id + 1) < len(ts_simulated_animal):
                next_ts = ts_simulated_animal[id + 1]
                try:
                    print(next_ts[0])
                    this_range = (df.index > this_ts) & (df.index < next_ts[0])
                except:
                    this_range = (df.index > this_ts) & (df.index < next_ts)
            else:
                print(df.iloc[len(df) - 1, :])
                this_range = (df.index > this_ts) & (df.index < df.index[len(df) - 1])
            fchop = str(this_ts)
        except:
            this_range = (df.index > this_ts[0]) & (
                df.index < this_ts[len(this_ts) - 1]
            )
            fchop = str(this_ts[0]).split(".")[0]
        heading_direction = df["SensRotY"][this_range]
        x = df["SensPosX"][this_range]
        # x = bfill(x.to_numpy())
        y = df["SensPosY"][this_range]
        xy = np.vstack((x.to_numpy(), y.to_numpy()))
        xy = bfill(xy)
        # y = bfill(y.to_numpy())
        ts = df["Current Time"][this_range]
        trial_no = df["CurrentTrial"][this_range]
        print(trial_no)
        if len(trial_no.value_counts()) > 1:
            return (
                heading_direction_across_trials,
                x_across_trials,
                y_across_trials,
                ts_across_trials,
            )

        if generate_locust_vr_matrices:
            ## Needs to tune this part later to make this work. And then we probably dont need bfill nan anymore. probably
            # loss, X, Y = removeNoiseVR(x.to_numpy(),y.to_numpy())
            # loss = 1 - loss
            loss = 0
            rX, rY = rotate_vector(xy[0], xy[1], -conditions[id]["Mu"] * np.pi / 180)
            newindex = diskretize(rX, rY, BODY_LENGTH)
            dX = np.array([rX[i] for i in newindex]).T
            dY = np.array([rY[i] for i in newindex]).T
            angles = heading_direction[newindex].to_numpy() * np.pi / 180
            # angles = np.array(ListAngles(dX, dY))
            c = np.cos(angles)
            s = np.sin(angles)
            xm = np.sum(c) / len(angles)
            ym = np.sum(s) / len(angles)

            meanAngle = atan2(ym, xm)
            meanVector = np.sqrt(np.square(np.sum(c)) + np.square(np.sum(s))) / len(
                angles
            )

            std = np.sqrt(2 * (1 - meanVector))

            tdist = len(dX) * BODY_LENGTH

            sin = meanVector * np.sin(meanAngle)
            cos = meanVector * np.cos(meanAngle)

            f = [fchop] * len(dX)
            loss = [loss] * len(dX)
            o = [conditions[id]["Kappa"]] * len(dX)
            d = [conditions[id]["Density"]] * len(dX)
            G = [growth_condition] * len(dX)

            df_curated = pd.DataFrame(
                {
                    "X": dX,
                    "Y": dY,
                    "fname": f,
                    "loss": loss,
                    "order": o,
                    "density": d,
                    "groups": G,
                }
            )
            f = [f[0]]
            loss = [loss[0]]
            o = [o[0]]
            d = [d[0]]
            G = [G[0]]
            V = [meanVector]
            MA = [meanAngle]
            ST = [std]
            lX = [dX[-1]]
            tD = [tdist]
            sins = [sin]
            coss = [cos]

            df_summary = pd.DataFrame(
                {
                    "fname": f,
                    "loss": loss,
                    "order": o,
                    "density": d,
                    "groups": G,
                    "mean_angle": MA,
                    "vector": V,
                    "variance": ST,
                    "distX": lX,
                    "distTotal": tD,
                    "sin": sins,
                    "cos": coss,
                }
            )
            with lock:
                store = pd.HDFStore(curated_file_path)
                store.append(
                    "name_of_frame",
                    df_curated,
                    format="t",
                    data_columns=df_curated.columns,
                )
                store.close()
                store = pd.HDFStore(summary_file_path)
                store.append(
                    "name_of_frame",
                    df_summary,
                    format="t",
                    data_columns=df_summary.columns,
                )
                store.close()
        heading_direction_across_trials.append(heading_direction)
        x_across_trials.append(x)
        y_across_trials.append(y)
        ts_across_trials.append(ts)
    return (
        heading_direction_across_trials,
        x_across_trials,
        y_across_trials,
        ts_across_trials,
    )


"""
might be useful in the future
    # ts = pd.to_datetime(df["Current Time"], format="%Y-%m-%d %H_%M_%S")
    # df["step_distance"] = np.sqrt(
    #     (df["SensPosX"].diff()) ** 2 + (df["SensPosY"].diff()) ** 2
    # )
    # heading_direction = df["SensRotY"]
    # df["step_distance_mm"] = df["step_distance"] * track_ball_radius
    # # calculate time between each step with Current Time: Timestamp('2024-05-16 14:16:35.300000')
    # df["time_diff"] = df["Current Time"].diff()
    # df["time_diff_ms"] = df["time_diff"].dt.total_seconds() * 1000
    # # calculate speed of each step
    # df["speed_mm_s"] = df["step_distance_mm"] / df["time_diff_ms"] * 1000
"""


def preprocess_matrex_data(thisDir, json_file):
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())
    num_vr = 4
    for i in range(num_vr):
        vr_pattern = f"*SimulatedLocustsVR{i+1}*"
        found_result = find_file(thisDir, vr_pattern)
        if found_result is None:
            return print(f"file with {vr_pattern} not found")
        else:
            ts_simulated_animal = []
            x_simulated_animal = []
            y_simulated_animal = []
            conditions = []
            if isinstance(found_result, list):
                print(
                    f"Analyze {vr_pattern} data which come with multiple trials of vr models. Use a for-loop to go through them"
                )

                for this_file in found_result:
                    ts, x, y, condition = read_simulated_data(
                        this_file, analysis_methods
                    )
                    ts_simulated_animal.append(ts)
                    x_simulated_animal.append(x)
                    y_simulated_animal.append(y)
                    conditions.append(condition)

            elif len(found_result.stem) > 0:
                ts, x, y, condition = read_simulated_data(
                    found_result, analysis_methods
                )
                ts_simulated_animal.append(ts)
                x_simulated_animal.append(x)
                y_simulated_animal.append(y)
                conditions.append(condition)

        locust_pattern = f"*_VR{i+1}*"
        found_result = find_file(thisDir, locust_pattern)
        if found_result is None:
            return print(f"file with {locust_pattern} not found")
        else:
            if isinstance(found_result, list):
                print(
                    f"Analyze {locust_pattern} data which come with multiple trials of vr models. Use a for-loop to go through them"
                )
                for this_file in found_result:
                    (
                        heading_direction_focal_animal,
                        x_focal_animal,
                        y_focal_animal,
                        ts_focal_animal,
                    ) = align_matrex_data(
                        this_file,
                        analysis_methods,
                        ts_simulated_animal,
                        x_simulated_animal,
                        y_simulated_animal,
                        conditions,
                    )
            elif len(found_result.stem) > 0:
                (
                    heading_direction_focal_animal,
                    x_focal_animal,
                    y_focal_animal,
                    ts_focal_animal,
                ) = align_matrex_data(
                    found_result,
                    analysis_methods,
                    ts_simulated_animal,
                    x_simulated_animal,
                    y_simulated_animal,
                    conditions,
                )


if __name__ == "__main__":
    thisDir = r"C:\Users\neuroPC\Documents\20240818_134521"
    json_file = r"C:\Users\neuroPC\Documents\GitHub\UnityDataAnalysis\analysis_methods_dictionary.json"
    json_file = {
        "overwrite_curated_dataset": True,
        "body_length": 0.12,
        "growth_condition": "G",
        "generate_locust_vr_matrices": True,
    }
    tic = time.perf_counter()
    preprocess_matrex_data(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")

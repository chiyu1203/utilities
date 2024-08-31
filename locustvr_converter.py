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
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import savgol_filter

"""
demo if you want to load packages from other directories
current_working_directory = Path.cwd()
parent_dir = current_working_directory.resolve().parents[0]
sys.path.insert(0, str(parent_dir) + "\\utilities")

"""


class MplColorHelper:
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)


lock = Lock()
colormap_name = "coolwarm"
COL = MplColorHelper(colormap_name, 0, 8)
sm = cm.ScalarMappable(cmap=colormap_name)


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
    if type(this_file) == str:
        this_file = Path(this_file)
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


def analyse_focal_animal(
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
    alpha_dictionary = {0.1: 0.2, 1.0: 0.4, 10.0: 0.6, 100000.0: 1}
    analyze_one_session_only = True
    BODY_LENGTH = analysis_methods.get("body_length")
    growth_condition = analysis_methods.get("growth_condition")
    generate_locust_vr_matrices = analysis_methods.get("generate_locust_vr_matrices")
    overwrite_curated_dataset = analysis_methods.get("overwrite_curated_dataset")
    heading_direction_across_trials = []
    x_across_trials = []
    y_across_trials = []
    ts_across_trials = []
    if type(this_file) == str:
        this_file = Path(this_file)
    if this_file.suffix == ".gz":
        with gzip.open(this_file, "rb") as f:
            df = pd.read_csv(f)
    elif this_file.suffix == ".csv":
        with open(this_file, mode="r") as f:
            df = pd.read_csv(f)
    # print(df.columns)
    df["GameObjectPosX"].replace(0.0, np.nan, inplace=True)
    df["GameObjectPosZ"].replace(0.0, np.nan, inplace=True)
    df["GameObjectRotY"].replace(0.0, np.nan, inplace=True)
    # sens pos is the data from fictrac
    # df["SensPosX"].replace(0.0, np.nan, inplace=True)
    # df["SensPosY"].replace(0.0, np.nan, inplace=True)
    df["SensRotY"].replace(0.0, np.nan, inplace=True)
    df["Current Time"] = pd.to_datetime(
        df["Current Time"], format="%Y-%m-%d %H:%M:%S.%f"
    )
    experiment_id = df["VR"][0] + " " + str(df["Current Time"][0]).split(".")[0]
    experiment_id = re.sub(r"\s+", "_", experiment_id)
    experiment_id = re.sub(r":", "", experiment_id)
    curated_file_path = this_file.parent / f"{experiment_id}_XY.h5"
    summary_file_path = this_file.parent / f"{experiment_id}_score.h5"
    if analysis_methods.get("analyse_turning_behaviour") == True:
        trim_seconds = 1.0
        df["elapsed_time"] = (
            df["Current Time"] - df["Current Time"].min()
        ).dt.total_seconds()
        grouped = df.groupby(["CurrentTrial", "CurrentStep"])
        tmp = grouped.apply(
            lambda x: x[
                (x["elapsed_time"] >= trim_seconds)
                & (x["elapsed_time"] <= x["elapsed_time"].max() - trim_seconds)
            ]
        ).reset_index(drop=True)
        tmp["dif_orientation"] = tmp["SensRotY"].diff()
        tmp_grouped = tmp.groupby(["CurrentTrial", "CurrentStep"])
        tmp["cumsum"] = tmp_grouped["dif_orientation"].transform(pd.Series.cumsum)

        # df.groupby(["CurrentTrial", "CurrentStep"])["dif_orientation"].cumsum()
        for name, entries in tmp_grouped:
            plt.plot
            # entries["dif_orientation"] = entries["SensRotY"].diff()
            # print(entries["dif_orientation"].cumsum().head(10))
            print(f'First 2 entries for the "{name}" category:')
            print(30 * "-")
            print(entries["cumsum"].head(5), "\n\n")
            print(entries["SensRotY"].head(5), "\n\n")

    if overwrite_curated_dataset == True and curated_file_path.is_file():
        curated_file_path.unlink()
        try:
            summary_file_path.unlink()
        except OSError as e:
            # If it fails, inform the user.
            print("Error: %s - %s." % (e.filename, e.strerror))
    if analysis_methods.get("plotting_trajectory") == True:
        fig, (ax1, ax2) = plt.subplots(
            nrows=1, ncols=2, figsize=(18, 7), tight_layout=True
        )
        ax1.set_title("ISI")
        ax2.set_title("Trial")
    for id in range(len(conditions)):
        this_range = (df["CurrentStep"] == id) & (df["CurrentTrial"] == 0)
        this_current_time = df["Current Time"][this_range]
        if len(this_current_time) == 0:
            break
        fchop = str(this_current_time.iloc[0]).split(".")[0]
        fchop = re.sub(r"\s+", "_", fchop)
        fchop = re.sub(r":", "", fchop)
        heading_direction = df["GameObjectRotY"][this_range]
        x = df["GameObjectPosX"][this_range]
        y = df["GameObjectPosZ"][this_range]
        xy = np.vstack((x.to_numpy(), y.to_numpy()))
        xy = bfill(xy)
        ts = df["Current Time"][this_range]
        trial_no = df["CurrentTrial"][this_range]
        print(trial_no)
        if len(trial_no.value_counts()) > 1 & analyze_one_session_only == True:
            break
        if generate_locust_vr_matrices:
            ## Needs to tune this part later to make this work. And then we probably dont need bfill nan anymore. probably

            # fig, (ax1, ax2) = plt.subplots(
            #     nrows=1, ncols=2, figsize=(18, 7), tight_layout=True
            # )
            # ax1.set_title("raw trace")
            # ax2.set_title("spacial discritisation")
            # ax1.plot(xy[0], xy[1])
            # trajectory_fig_path = (
            #     this_file.parent / f"{experiment_id}_trajectory_{id}.png"
            # )
            # fig.savefig(trajectory_fig_path)

            if id == 3:
                print("ready to fight")
            loss, X, Y = removeNoiseVR(xy[0], xy[1])
            loss = 1 - loss
            if len(X) == 0:
                print("all is noise")
                continue
            rX, rY = rotate_vector(X, Y, -90 * np.pi / 180)
            newindex = diskretize(rX, rY, BODY_LENGTH)
            dX = np.array([rX[i] for i in newindex]).T
            dY = np.array([rY[i] for i in newindex]).T
            # ax2.plot(dX, dY)
            # trajectory_fig_path = (
            #     this_file.parent / f"{experiment_id}_trajectory_{id}.png"
            # )
            # fig.savefig(trajectory_fig_path)
            angles = np.array(ListAngles(dX, dY))
            # angles = heading_direction[newindex].to_numpy() * np.pi / 180
            # test = heading_direction.to_numpy()
            # angle1 = np.array([(test[i]) * np.pi / 180 for i in newindex]).T
            # plt.scatter(np.arange(angle1.shape[0]), angle1, c="b")
            # plt.scatter(np.arange(angles.shape[0]), angles, c="r")
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
            mu = [conditions[id]["Mu"]] * len(dX)
            spe = [conditions[id]["LocustSpeed"]] * len(dX)

            G = [growth_condition] * len(dX)

            df_curated = pd.DataFrame(
                {
                    "X": dX,
                    "Y": dY,
                    "fname": f,
                    "loss": loss,
                    "order": o,
                    "density": d,
                    "mu": mu,
                    "agent_speed": spe,
                    "groups": G,
                }
            )
            f = [f[0]]
            loss = [loss[0]]
            o = [o[0]]
            d = [d[0]]
            mu = [mu[0]]
            spe = [spe[0]]
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
                    "mu": mu,
                    "agent_speed": spe,
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
        if analysis_methods.get("analyse_turning_behaviour") == True:
            print("analyse turning behaviour based on jaw vector")
            # xy = bfill(xy)
            # ts = df["Current Time"][this_range]
            # trial_no = df["CurrentTrial"][this_range]
            # heading_direction = df["GameObjectRotY"][this_range]
        if analysis_methods.get("filtering_method") == "sg_filter":
            x_all = savgol_filter(xy[0], 71, 3, axis=0)
            y_all = savgol_filter(xy[1], 71, 3, axis=0)
        else:
            x_all = xy[0]
            y_all = xy[1]
        elapsed_time = (ts - ts.min()).dt.total_seconds()
        df["elapsed_time"] = (
            df["Current Time"] - df["Current Time"].min()
        ).dt.total_seconds()

        if analysis_methods.get("plotting_trajectory") == True:
            if df_summary["density"][0] > 0:
                # ax2.plot(
                #     dX, dY, color=np.arange(len(dY)), alpha=df_curated.iloc[id]["alpha"]
                # )
                ##blue is earlier colour and yellow is later colour
                ax2.scatter(
                    dX,
                    dY,
                    c=np.arange(len(dY)),
                    marker=".",
                    alpha=df_summary["order"].map(alpha_dictionary)[0],
                )
            else:
                # ax1.plot(
                #     dX, dY, alpha=df_curated.iloc[id]["alpha"]
                # )
                ax1.scatter(
                    dX,
                    dY,
                    c=np.arange(len(dY)),
                    marker=".",
                    alpha=df_summary["order"].map(alpha_dictionary)[0],
                )

        heading_direction_across_trials.append(heading_direction)
        x_across_trials.append(x)
        y_across_trials.append(y)
        ts_across_trials.append(ts)
    trajectory_fig_path = this_file.parent / f"{experiment_id}_trajectory.png"
    if analysis_methods.get("plotting_trajectory") == True:
        fig.savefig(trajectory_fig_path)
    return (
        heading_direction_across_trials,
        x_across_trials,
        y_across_trials,
        ts_across_trials,
    )


"""
used to align dataset
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
"""


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


    
    if dir_list[0]==dir_list[1]:
        for this_dir in dir_list[::num_vr]:
            h5_dirs=find_file(this_dir,h5_pattern)
            fig = plt.figure(figsize=(18, 5),tight_layout=True)
            ax1 = plt.subplot2grid((1, 18), (0, 0),colspan=8)
            ax2 = plt.subplot2grid((1, 18), (0, 8))
            ax3 = plt.subplot2grid((1, 18), (0, 9),colspan=8)
            ax4 = plt.subplot2grid((1, 18), (0, 17))
            for idx,this_file in enumerate(h5_dirs):
                this_color=colour_code[idx]
                if this_file.stem in ['VR4_Swarm_2024-08-16_131719_score','VR4_Swarm_2024-08-16_145857_score']:
                    continue
                df = pd.read_hdf(this_file)
                df_stim = df.loc[(df['loss'] < 0.05) & (df['distTotal'] >= 12)&(df ['density'] > 0)] 
                df_stim = df_stim.reset_index(drop=True)
                ax1.set_xscale('log')
                ax1.set_ylim([-4,4])        
                ax3.set_xscale('log')
                ax3.set_ylim([1,2000])
                ax1.scatter(df_stim['order'], df_stim['mean_angle'],c=this_color)
                ax3.scatter(df_stim['order'], df_stim['distTotal'],c=this_color)
                ax2.set_ylim([-4,4])
                ax2.set_yticks([])
                ax2.set_xticks([])
                ax4.set_ylim([1,1000])
                ax4.set_yticks([])
                ax4.set_xticks([])
                df_isi = df.loc[(df['loss'] < 0.05) & (df['distTotal'] >= 12)&(df ['density'] == 0)]
                df_isi = df_isi.reset_index(drop=True)
                if len(df_isi)>0:
                    ax2.scatter(df_isi.iloc[0]['order']/2, df_isi.iloc[0]['mean_angle'],c=this_color)
                    #ax2.scatter(df.iloc[-1]['order'], df.iloc[-1]['mean_angle'],c=this_color,alpha=0.2)
                    ax4.scatter(df_isi.iloc[0]['order']/2, df_isi.iloc[0]['distTotal'],c=this_color)
                    #ax4.scatter(df.iloc[-1]['order'], df.iloc[-1]['distTotal'],c=this_color,alpha=0.2)    


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
        # i = i + 3
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
                    ) = analyse_focal_animal(
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
                ) = analyse_focal_animal(
                    found_result,
                    analysis_methods,
                    ts_simulated_animal,
                    x_simulated_animal,
                    y_simulated_animal,
                    conditions,
                )


if __name__ == "__main__":
    # thisDir = r"D:\MatrexVR_Swarm_Data\RunData\20240818_170807"
    thisDir = r"D:\MatrexVR_Swarm_Data\RunData\20240826_150826"
    json_file = r"C:\Users\neuroPC\Documents\GitHub\UnityDataAnalysis\analysis_methods_dictionary.json"
    json_file = {
        "overwrite_curated_dataset": True,
        "plotting_trajectory": True,
        "body_length": 12,
        "growth_condition": "G",
        "generate_locust_vr_matrices": True,
        "analyse_turning_behaviour": False,
        "filtering_method": "sg_filter",
    }
    tic = time.perf_counter()
    preprocess_matrex_data(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")

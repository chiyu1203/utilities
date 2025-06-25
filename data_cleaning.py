import pandas as pd
import numpy as np
import os, math, re, csv, json, sys
import time
from math import atan2
import matplotlib.pyplot as plt
# import chardet
from pathlib import Path
from useful_tools import find_file
from scipy.interpolate import interp1d
# def ListAngles(X,Y):
#     ang = [] 
#     for i in range(len(X)-1):
#         changeInX = X[i+1] - X[i]
#         changeInY = Y[i+1] - Y[i]
        
#         a = atan2(changeInY,changeInX)
#         if a < 0:
#             a = a + 2*np.pi
#         ang.append(a)
        
#     return ang

# def calc_eucledian(xx1,yy1):
#     w=0
#     dist = []
#     while w < len(xx1)-1:
        

#         a = np.array((xx1[w+1],yy1[w+1]))
#         b = np.array((xx1[w],yy1[w]))        

#         euc = np.linalg.norm(a - b)
#         #euc = distance.euclidean(b,a)
#         dist.append(euc)
#         w = w + 1

#     return dist

def interp_fill(arr):
    arr = arr.copy()
    if arr.ndim == 1:
        return _interp_1d(arr)
    elif arr.ndim == 2:
        return np.apply_along_axis(_interp_1d, axis=1, arr=arr)

def _interp_1d(x):
    window_size=3
    n = len(x)
    isnan = np.isnan(x)
    if not isnan.any():
        return x

    valid_idx = np.where(~isnan)[0]
    valid_vals = x[valid_idx]

    # If fewer than 3 values exist, use basic interpolation
    if len(valid_idx) < window_size:
        return np.interp(np.arange(n), valid_idx, valid_vals)

    result = x.copy()

    # Interpolate inside using up to 3 nearest neighbors
    for i in np.where(isnan)[0]:
        # Find nearest 3 neighbors (could be before or after)
        nearby = valid_idx[np.argsort(np.abs(valid_idx - i))][:window_size]
        nearby_vals = x[nearby]

        # Sort for interpolation
        sorted_idx = np.sort(nearby)
        sorted_vals = x[sorted_idx]

        # Linear interpolation
        f = interp1d(sorted_idx, sorted_vals, kind='linear', fill_value="extrapolate")
        result[i] = f(i)

    return result


def findLongestConseqSubseq(arr, n):
    """We insert all the array elements into unordered set. from https://www.geeksforgeeks.org/maximum-consecutive-numbers-present-array/"""
    S = set()
    for i in range(n):
        S.add(arr[i])

    # check each possible sequence from the start
    # then update optimal length
    ans = 0
    for i in range(n):

        # if current element is the starting
        # element of a sequence
        if S.__contains__(arr[i]):

            # Then check for next elements in the
            # sequence
            j = arr[i]

            # increment the value of array element
            # and repeat search in the set
            while S.__contains__(j):
                j += 1

            # Update optimal length if this length
            # is more. To get the length as it is
            # incremented one by one
            ans = max(ans, j - arr[i])
    return ans


def diskretize(
    x, y, bodylength
):  # discretize data into equidistant points, using body length (https://stackoverflow.com/questions/19117660/how-to-generate-equispaced-interpolating-values)
    # code writen by Sercan Sayin and described in (https://www.science.org/doi/10.1126/science.adq7832)
    # the source code can be found in (https://zenodo.org/records/14355590)
    tol = bodylength  # 12cm ,roughly 3BL
    i, idx = 0, [0]
    while i < len(x) - 1:
        total_dist = 0
        for j in range(i + 1, len(x)):
            total_dist = math.sqrt((x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2)
            if total_dist > tol:
                idx.append(j)
                break
        i = j + 1

    return idx


def ffill(arr):
    mask = np.isnan(arr)
    if arr.ndim == 1:
        idx = np.where(~mask, np.arange(mask.shape[0]), 0)
        np.maximum.accumulate(idx, out=idx)
        out = arr[idx]
    elif arr.ndim == 2:
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        out = arr[np.arange(idx.shape[0])[:, None], idx]
    return out


# Simple solution for bfill provided by financial_physician in comment below
#https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
def bfill(arr):
    if arr.ndim == 1:
        return ffill(arr[::-1])[::-1]
    elif arr.ndim == 2:
        return ffill(arr[:, ::-1])[:, ::-1]


def remove_unreliable_tracking(X, Y, analysis_methods):
    using_heading_mask=True
    time_series_analysis = analysis_methods.get("time_series_analysis")
    exp_name=analysis_methods.get("experiment_name")
    travel_distance_fbf = euclidean_distance(X,Y)
    if exp_name.lower() == "locustvr":
        if time_series_analysis:
            noise_index = np.argwhere(travel_distance_fbf > 1.0)
            mask=noise_index.T
            X[mask] = np.nan
            Y[mask] = np.nan
            good_track_ratio = (len(X) - mask.shape[1]) / len(X)
            if good_track_ratio<1:
                print("let's check")
        else:
            noise_index = np.argwhere(travel_distance_fbf > 1.0)
            Xraw = X        
            dX = np.diff(X)
            dY = np.diff(Y)
            if using_heading_mask:
                angles = np.arctan2(dY, dX)
                a = np.abs(np.gradient(angles)) 
                indikes = np.argwhere(a< 0.0001) 
                mask1 = np.ones_like(dX, dtype=bool)
                mask1[noise_index] = False
                mask2 = np.ones_like(dX, dtype=bool)
                mask2[indikes]=False
                mask=mask1 & mask2
            else:
                mask = np.ones_like(dX, dtype=bool)
                mask[noise_index] = False
            X = np.nancumsum(dX[mask])
            Y = np.nancumsum(dY[mask])
            #### there seems to be a bug from the funcs.py removeNoiseVR(X,Y,ts=None)
            # a = np.array(calc_eucledian(X,Y))
            # indikes = np.argwhere( a > 0.01)
            
            # NewX = np.delete(np.diff(X), indikes.T)       
            # NewY = np.delete(np.diff(Y), indikes.T) 
            # X = np.cumsum(NewX)
            # Y = np.cumsum(NewY)
            # a = ListAngles(X,Y)
            # a = np.abs(np.gradient(a))
            # indikes = np.argwhere( a < 0.0001) 
        
            # NewX = np.delete(np.diff(X), indikes.T)       
            # NewY = np.delete(np.diff(Y), indikes.T) 
            # X = np.cumsum(NewX)
            # Y = np.cumsum(NewY)
            good_track_ratio = len(X)/len(Xraw)
    else:
        noise_index = np.argwhere(
                travel_distance_fbf > 0.4
            )#arbitary threshold based on several videos in Swarm scene
        # fig, ax1 = plt.subplots(1, 1, figsize=(18, 7), tight_layout=True)
        # ax1.hist(travel_distance_fbf)
        # fig.savefig('sanity_check.png')
        if time_series_analysis:
            # noise_index = np.argwhere(travel_distance_fbf > 1.0)
            mask=noise_index.T
            """
            arbitary threshold based on VR4_2024-11-16_155210: 0.3 can remove fictrac bad tracking, 
            but at the risk of removing running data. 1.0 is safer or 
            follow what I did with bonfic data, extract potential epochs with 0.5 for 20 frames and 
            then apply fft to get auc value during <1 Hz"""
            X[mask] = np.nan
            Y[mask] = np.nan
            good_track_ratio = (len(X) - mask.shape[1]) / len(X)
        else:
            Xraw = X
            dX = np.diff(X)
            dY = np.diff(Y)
            mask = np.ones_like(travel_distance_fbf, dtype=bool)
            mask[noise_index] = False
            X = np.nancumsum(dX[mask])
            Y = np.nancumsum(dY[mask])
            good_track_ratio = len(X) / len(Xraw)
    return good_track_ratio, X, Y,mask

def euclidean_distance(X,Y):
    return np.sqrt(np.add(np.square(np.diff(X)), np.square(np.diff(Y))))

def update_csv_value(old_csv_path, new_csv_path):
    with open(old_csv_path, "r") as x:
        old_csv = list(csv.reader(x, delimiter=","))
        # old_csv = np.genfromtxt(x,delimiter=",")
    old_csv = np.genfromtxt(old_csv_path, delimiter=",")
    new_csv = np.genfromtxt(new_csv_path, delimiter=",", dtype=int)
    old_csv[:, 0] = new_csv
    np.savetxt(old_csv_path, old_csv, delimiter=",", fmt="%1.4f")
    return None


def update_csv_value_pd(old_csv_path, new_csv_path, old_column_number):
    tmp = pd.read_csv(old_csv_path, delimiter=",", header=0, index_col=None)
    new_csv = np.genfromtxt(new_csv_path, delimiter=",", dtype=int)
    if "Item2" in tmp.columns:
        new = tmp["Item2"].str.split(";", n=1, expand=True)
        tmp["Temperature"] = new[0]
        tmp["Humidity"] = new[1]
        tmp.drop(columns=["Item2"], inplace=True)
        # tmp["Item2"]=tmp["Item2"].str.split(";",expand=False)
        # tmp["Item2"]=tmp["Item2"].str.replace(";",",")
        tmp.reset_index(inplace=True)
        tmp.rename(columns={"index": "Value", "Item1": "Timestamp"}, inplace=True)

        if new_csv.shape[0] > tmp.shape[0]:
            new_csv = new_csv[: tmp.shape[0]]
        elif new_csv.shape[0] < tmp.shape[0]:
            test = np.ones(tmp.shape[0] - new_csv.shape[0])
            new_csv = np.insert(new_csv, obj=0, values=test)
        else:
            pass
    else:
        pass
    tmp.iloc[:, old_column_number] = new_csv
    tmp.to_csv(old_csv_path, index=False)
    return None


def load_temperature_data(txt_path):
    if type(txt_path) == str:
        txt_path = Path(txt_path)
    if txt_path.suffix == ".txt":
        # data comes from EL-USB
        # instead of using the first column as index, use the second column to log in index as dateindex. This is easier for resample
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html
        df = pd.read_csv(
            txt_path,
            parse_dates=[1],
            header=0,
            sep=",",
            index_col=1,
            encoding="unicode_escape",
            engine="python",
        )
        df.drop(df.columns[[0, 4]], axis=1, inplace=True)
        # then maybe use this method .interpolate()
        df = df.resample("1s").interpolate()
        # this for microsecond"1U"
        # this for milliseonds"L"
        # print(df.dtypes)

    elif txt_path.suffix == ".csv":
        if str(txt_path.stem).startswith("DL220THP"):
            df = pd.read_csv(
                txt_path,
                parse_dates=[0],
                dayfirst=True,
                skiprows=8,
                index_col=0,
                header=0,
                sep=",",
            )
            df.drop(df.columns[3], axis=1, inplace=True)
            df = df.resample("1s").interpolate()
        else:
            print("Here to process data loaded in Bonsai")

    return df


def sorting_trial_info(stim_info, analysis_methods,exp_date="XXXXXX"):
    visual_paradigm_name = analysis_methods.get("experiment_name")
    exp_place = analysis_methods.get("exp_place")
    camera_fps=analysis_methods.get("camera_fps")
    pre_stim_interval = analysis_methods.get("prestim_duration")
    stim_info = stim_info.reset_index()
    first_event_time=stim_info.iloc[0,0]/camera_fps
    update_pre_stim_interval=False
    if first_event_time>pre_stim_interval and update_pre_stim_interval:
        analysis_methods.update({"prestim_duration":first_event_time})
    raw_column_number = stim_info.shape[1]
    if visual_paradigm_name.lower() == "gratings" and exp_place.lower() != "zball":
        stim_variable = []
        stimulus_timestamp = []
        for row in range(0, len(stim_info)):

            # delete the bracket in current speed and save it as integer in new list
            stim_variable.append(int(stim_info.iloc[row, 2].replace("(", "")))
            stimulus_timestamp.append(float(stim_info.iloc[row, 3].replace(")", "")))
        stim_variable = np.array(stim_variable).reshape((len(stim_info), -1))
        stim_info["stim_type"] = stim_variable
        stim_info["ts"] = stimulus_timestamp
        stim_info.drop(columns=stim_info.columns[0:raw_column_number], inplace=True)
    elif visual_paradigm_name.lower() == "sweeploom":
        stim_variable = []
        stimulus_timestamp = []
        stim_type = analysis_methods.get("stim_type",[
            "c2b_f_sloom",
            "c2b_s_sloom",
            "c2f_f_sloom",
            "c2f_s_sloom",
            "c2c_f_loom",
            "c2c_s_loom",
        ])
        for row in range(0, len(stim_info)):
            stim_variable.append(int(stim_info.iloc[row, 2].split("=", 1)[1]))
            stim_variable.append(int(stim_info.iloc[row, 3].split("=", 1)[1]))
            stim_variable.append(int(stim_info.iloc[row, 4].split("=", 1)[1]))
            stim_variable.append(int(stim_info.iloc[row, 5].split("=", 1)[1]))
            stim_variable.append(int(stim_info.iloc[row, 6].split("=", 1)[1]))
            if exp_date in ["221102", "221103"]:
                stim_variable.append(int(stim_info.iloc[row, 7].split("=", 1)[1]))
                stim_variable.append(int(stim_info.iloc[row, 8].split("=", 1)[1]))
            # stim_variable.append(int(re.split(r'=|}',stim_info.iloc[row,5])[1]))
            if isinstance(stim_info.loc[row, "Value"], float):
                timestamp = stim_info.loc[row, "Timestamp"]
            else:
                timestamp = (
                    str(stim_info.loc[row, "Timestamp"])
                    + "."
                    + stim_info.loc[row, "Value"]
                )
            stim_variable.append(float(timestamp.replace(")", "")))
        stim_variable = np.array(stim_variable).reshape((len(stim_info), -1))
        column_names = [
            "size_start",
            "size_end",
            "location1_start",
            "location1_end",
            "location2_start",
            "location2_end",
            "time",
            "timestamp",
        ]
        stim_info = pd.DataFrame(stim_variable, columns=column_names)
        if exp_date in ["2022-10-15", "2022-10-16"]:
            ### use this criteria for experiments during exp_date=['221015','221016']
            filters = [
                (stim_info.iloc[:, 0] == 1)
                & (stim_info.iloc[:, 2] > stim_info.iloc[:, 3]),
                (stim_info.iloc[:, 0] == 8)
                & (stim_info.iloc[:, 2] > stim_info.iloc[:, 3]),
                (stim_info.iloc[:, 0] == 1)
                & (stim_info.iloc[:, 2] < stim_info.iloc[:, 3]),
                (stim_info.iloc[:, 0] == 8)
                & (stim_info.iloc[:, 2] < stim_info.iloc[:, 3]),
                (stim_info.iloc[:, 0] == 1)
                & (stim_info.iloc[:, 2] == stim_info.iloc[:, 3]),
                (stim_info.iloc[:, 0] == 8)
                & (stim_info.iloc[:, 2] == stim_info.iloc[:, 3]),
            ]

        else:
            filters = [
                (stim_info.iloc[:, 6] == 1)
                & (stim_info.iloc[:, 2] > stim_info.iloc[:, 3]),
                (stim_info.iloc[:, 6] == 2)
                & (stim_info.iloc[:, 2] > stim_info.iloc[:, 3]),
                (stim_info.iloc[:, 6] == 1)
                & (stim_info.iloc[:, 2] < stim_info.iloc[:, 3]),
                (stim_info.iloc[:, 6] == 2)
                & (stim_info.iloc[:, 2] < stim_info.iloc[:, 3]),
                (stim_info.iloc[:, 6] == 1)
                & (stim_info.iloc[:, 2] == stim_info.iloc[:, 3]),
                (stim_info.iloc[:, 6] == 2)
                & (stim_info.iloc[:, 2] == stim_info.iloc[:, 3]),
            ]
        stim_info["stim_type"] = np.select(filters, stim_type)
    elif visual_paradigm_name.lower() == "conflict" or visual_paradigm_name.lower() == "looming" or visual_paradigm_name.lower() == "receding" or visual_paradigm_name.lower() == "gratings":
        stim_variable = []
        stimulus_timestamp = []
        for row in range(0, np.shape(stim_info)[0]):
            for col in range(2, np.shape(stim_info)[1] - 1):
                if col == np.shape(stim_info)[1] - 2:
                    stim_variable.append(
                        int(stim_info.iloc[row, col].split("=", 1)[1].split("}", 1)[0])
                    )
                else:
                    stim_variable.append(
                        int(float(stim_info.iloc[row, col].split("=", 1)[1]))
                    )
            timestamp = stim_info.loc[row, "Value"]
            stim_variable.append(float(timestamp.replace(")", "")))
        stim_variable = np.array(stim_variable).reshape((len(stim_info), -1))
        default_column_names = [
            "LocationBeginX1","LocationEndX1","LocationBeginZ1","LocationEndZ1","PolarBeginR1","PolarEndR1","PolarBeginDegree1",
            "PolarEndDegree1","Phase1","PreMovDuration","Duration","PostMovDuration","ISI","LocustObj1","ReverseZ1","LocustTexture1","TranslationalGain1","RotationalGain1"]
        this_column_names=default_column_names[:stim_variable.shape[1]-1]
        this_column_names.append("ts")
        stim_info = pd.DataFrame(stim_variable, columns=this_column_names)
        cols_to_convert = ["LocustTexture1","ReverseZ1","LocustObj1","PolarBeginDegree1","PolarEndDegree1","Phase1","Duration"]           
        stim_info[cols_to_convert] = stim_info[cols_to_convert].astype(int)
        duration_sorted=sorted(stim_info["Duration"].unique())
        begin_degree_sorted=sorted(stim_info["PolarBeginDegree1"].unique())
        if visual_paradigm_name.lower() == "conflict":
            stim_type = [
                "cc_back_slow",
                "cc_back_medium",
                "cc_back_fast",
                "cc_front_slow",
                "cc_front_medium",
                "cc_front_fast",
                "c_back_slow",
                "c_back_medium",
                "c_back_fast",
                "c_front_slow",
                "c_front_medium",
                "c_front_fast",
            ]
            filters = [
            (stim_info["Duration"] == duration_sorted[2])
            & (stim_info["Phase1"] == 180)
            & (stim_info["PolarEndDegree1"] < stim_info["PolarBeginDegree1"]),
            (stim_info["Duration"] == duration_sorted[1])
            & (stim_info["Phase1"] == 180)
            & (stim_info["PolarEndDegree1"] < stim_info["PolarBeginDegree1"]),
            (stim_info["Duration"] == duration_sorted[0])
            & (stim_info["Phase1"] == 180)
            & (stim_info["PolarEndDegree1"] < stim_info["PolarBeginDegree1"]),
            (stim_info["Duration"] == duration_sorted[2])
            & (stim_info["Phase1"] == 0)
            & (stim_info["PolarEndDegree1"] < stim_info["PolarBeginDegree1"]),
            (stim_info["Duration"] == duration_sorted[1])
            & (stim_info["Phase1"] == 0)
            & (stim_info["PolarEndDegree1"] < stim_info["PolarBeginDegree1"]),
            (stim_info["Duration"] == duration_sorted[0])
            & (stim_info["Phase1"] == 0)
            & (stim_info["PolarEndDegree1"] < stim_info["PolarBeginDegree1"]),
            (stim_info["Duration"] == duration_sorted[2])
            & (stim_info["Phase1"] == 180)
            & (stim_info["PolarEndDegree1"] > stim_info["PolarBeginDegree1"]),
            (stim_info["Duration"] == duration_sorted[1])
            & (stim_info["Phase1"] == 180)
            & (stim_info["PolarEndDegree1"] > stim_info["PolarBeginDegree1"]),
            (stim_info["Duration"] == duration_sorted[0])
            & (stim_info["Phase1"] == 180)
            & (stim_info["PolarEndDegree1"] > stim_info["PolarBeginDegree1"]),
            (stim_info["Duration"] == duration_sorted[2])
            & (stim_info["Phase1"] == 0)
            & (stim_info["PolarEndDegree1"] > stim_info["PolarBeginDegree1"]),
            (stim_info["Duration"] == duration_sorted[1])
            & (stim_info["Phase1"] == 0)
            & (stim_info["PolarEndDegree1"] > stim_info["PolarBeginDegree1"]),
            (stim_info["Duration"] == duration_sorted[0])
            & (stim_info["Phase1"] == 0)
            & (stim_info["PolarEndDegree1"] > stim_info["PolarBeginDegree1"]),
        ]
            stim_info["stim_type"] = np.select(filters, stim_type,default="unclassified")
        elif visual_paradigm_name.lower() == "gratings":
            stim_info["stim_type"] = stim_info["PolarBeginDegree1"].astype(int)
        else:
            stim_type = [
                "receding_left_slow",
                "receding_center_slow",
                "receding_right_slow",
                "receding_left_medium",
                "receding_center_medium",
                "receding_right_medium",
                "receding_left_fast",
                "receding_center_fast",
                "receding_right_fast",
                "looming_left_slow",
                "looming_center_slow",
                "looming_right_slow",
                "looming_left_medium",
                "looming_center_medium",
                "looming_right_medium",
                "looming_left_fast",
                "looming_center_fast",
                "looming_right_fast",
            ]

            filters = [
            (stim_info["Duration"] == duration_sorted[2])
            & (stim_info["PolarBeginR1"] < stim_info["PolarEndR1"])
            &(stim_info["PolarEndDegree1"] == begin_degree_sorted[2]),
            (stim_info["Duration"] == duration_sorted[2])
            & (stim_info["PolarBeginR1"] < stim_info["PolarEndR1"])
            &(stim_info["PolarEndDegree1"] == begin_degree_sorted[1]),
            (stim_info["Duration"] == duration_sorted[2])
            & (stim_info["PolarBeginR1"] < stim_info["PolarEndR1"])
            &(stim_info["PolarEndDegree1"] == begin_degree_sorted[0]),
            (stim_info["Duration"] == duration_sorted[1])
            & (stim_info["PolarBeginR1"] < stim_info["PolarEndR1"])
            &(stim_info["PolarEndDegree1"] == begin_degree_sorted[2]),
            (stim_info["Duration"] == duration_sorted[1])
            & (stim_info["PolarBeginR1"] < stim_info["PolarEndR1"])
            &(stim_info["PolarEndDegree1"] == begin_degree_sorted[1]),
            (stim_info["Duration"] == duration_sorted[1])
            & (stim_info["PolarBeginR1"] < stim_info["PolarEndR1"])
            &(stim_info["PolarEndDegree1"] == begin_degree_sorted[0]),
            (stim_info["Duration"] == duration_sorted[0])
            & (stim_info["PolarBeginR1"] < stim_info["PolarEndR1"])
            &(stim_info["PolarEndDegree1"] == begin_degree_sorted[2]),
            (stim_info["Duration"] == duration_sorted[0])
            & (stim_info["PolarBeginR1"] < stim_info["PolarEndR1"])
            &(stim_info["PolarEndDegree1"] == begin_degree_sorted[1]),
            (stim_info["Duration"] == duration_sorted[0])
            & (stim_info["PolarBeginR1"] < stim_info["PolarEndR1"])
            &(stim_info["PolarEndDegree1"] == begin_degree_sorted[0]),
            (stim_info["Duration"] == duration_sorted[2])
            & (stim_info["PolarBeginR1"] > stim_info["PolarEndR1"])
            &(stim_info["PolarEndDegree1"] == begin_degree_sorted[2]),
            (stim_info["Duration"] == duration_sorted[2])
            & (stim_info["PolarBeginR1"] > stim_info["PolarEndR1"])
            &(stim_info["PolarEndDegree1"] == begin_degree_sorted[1]),
            (stim_info["Duration"] == duration_sorted[2])
            & (stim_info["PolarBeginR1"] > stim_info["PolarEndR1"])
            &(stim_info["PolarEndDegree1"] == begin_degree_sorted[0]),
            (stim_info["Duration"] == duration_sorted[1])
            & (stim_info["PolarBeginR1"] > stim_info["PolarEndR1"])
            &(stim_info["PolarEndDegree1"] == begin_degree_sorted[2]),
            (stim_info["Duration"] == duration_sorted[1])
            & (stim_info["PolarBeginR1"] > stim_info["PolarEndR1"])
            &(stim_info["PolarEndDegree1"] == begin_degree_sorted[1]),
            (stim_info["Duration"] == duration_sorted[1])
            & (stim_info["PolarBeginR1"] > stim_info["PolarEndR1"])
            &(stim_info["PolarEndDegree1"] == begin_degree_sorted[0]),
            (stim_info["Duration"] == duration_sorted[0])
            & (stim_info["PolarBeginR1"] > stim_info["PolarEndR1"])
            &(stim_info["PolarEndDegree1"] == begin_degree_sorted[2]),
            (stim_info["Duration"] == duration_sorted[0])
            & (stim_info["PolarBeginR1"] > stim_info["PolarEndR1"])
            &(stim_info["PolarEndDegree1"] == begin_degree_sorted[1]),
            (stim_info["Duration"] == duration_sorted[0])
            & (stim_info["PolarBeginR1"] > stim_info["PolarEndR1"])
            &(stim_info["PolarEndDegree1"] == begin_degree_sorted[0]),
        ]
            stim_info["stim_type"] = np.select(filters, stim_type,default="unclassified")
    else:
        col_index = [2, 3, 4, 5, 7, 11, 13, 15, 17]
        col_name = [
            "ID",
            "Duration",
            "Size",
            "Colour",
            "numDots",
            "Contrast",
            "VelX",
            "VelY",
            "Coherence",
        ]
        for i, j in zip(col_index, col_name):
            stim_info[["tmp", j]] = stim_info.iloc[:, i].str.split("=", expand=True)
            stim_info[j] = pd.to_numeric(stim_info[j])
        if exp_place.lower() == "zball" or exp_place.lower() == "matrexvr":
            stim_info["ts"] = stim_info["Value"].str.replace(r"[()]", "", regex=True)
        elif exp_place.lower() == "vccball":
            stim_info["Value"] = stim_info["Value"].str.replace(r"[()]", "", regex=True)
            if stim_info["Timestamp"].dtypes == "object":
                stim_info["Timestamp"] = stim_info["Timestamp"].str.replace(
                    r"[()]", "", regex=True
                )
            stim_info["Value"] = stim_info["Value"].fillna(0)
            stim_info["Timestamp"] = stim_info["Timestamp"].astype(str)
            stim_info["Value"] = stim_info["Value"].astype(str)
            stim_info["ts"] = stim_info[["Timestamp", "Value"]].agg(".".join, axis=1)
        stim_info["ts"] = pd.to_numeric(stim_info["ts"])
        stim_info["ts"] = stim_info["ts"].astype("float32")
        stim_info.drop(
            columns=stim_info.columns[0 : raw_column_number + 1], inplace=True
        )
        if visual_paradigm_name.endswith("sities"):
            stim_variable_direction = (
                stim_info["VelX"] * stim_info["numDots"] / abs(stim_info["VelY"])+abs(stim_info["VelX"])
            )
        elif visual_paradigm_name.lower() == "coherence":
            stim_variable_direction = (
                (stim_info["VelY"]+stim_info["VelX"]) * stim_info["Coherence"] / (abs(stim_info["VelY"])+abs(stim_info["VelX"]))
            )## add the plus here because when inserting the probe from anterior side, the dots move along the y axis
        # positive value means dots moves from right to left in a window, allocentric orientation in a panoramic setup: counterclock wise
        # if np.isnan(stim_variable_direction).any():
        #     stim_variable_direction=stim_variable_direction.fillna(0)
        stim_info["stim_type"] = -1 * stim_variable_direction.astype(int)
    stim_type = np.sort(stim_info.stim_type.unique()).tolist()
    return stim_info, stim_type

def generate_timestamp_csv(file_path):
    save_csv = False
    file_name, file_extension = os.path.splitext(file_path)
    output_file_name = file_name + "_timestamp.csv"
    if file_extension == ".log":
        frame_pattern = r"Trackball::process \[INF\] Frame (\d+)"
        float_pattern = r"BaslerSource::grab \[DBG\] Frame captured \d+x\d+x\d+ @ \d+\.\d+ \((\d+\.\d+)\)"

        # Detect the file's encoding
        # with open(file_path, "rb") as rawdata:
        #     result = chardet.detect(rawdata.read(10000))

        # detected_encoding = result["encoding"]

        # Open and read the log file (used above command and found out that encoding is ISO-8859-5 in this file)
        with open(file_path, "r", encoding="ISO-8859-5") as log_file:
            log_content = log_file.read()
        # Use regular expressions to extract frame numbers and float values
        frame_numbers = [int(match) for match in re.findall(frame_pattern, log_content)]
        float_values = [
            float(value) for value in re.findall(float_pattern, log_content)
        ]
    else:
        tracking_info = pd.read_csv(file_path, delimiter=",", engine="python")
        tracking_info["Timestamp"] = tracking_info["Timestamp"].str.replace(
            r"[()]", "", regex=True
        )
        tracking_info["Timestamp"] = pd.to_numeric(tracking_info["Timestamp"])
        tracking_info["Value"] = tracking_info["Value"].str.replace(
            r"[()]", "", regex=True
        )
        tracking_info["Value"] = pd.to_numeric(tracking_info["Value"])
        # frame_number = tracking_info["Value"].drop_duplicates()
        # fill_element = np.ones(frame_number.values[0])
        float_values = tracking_info["Timestamp"][
            tracking_info["Value"].drop_duplicates().index
        ].values
        # print(tracking_info)
        # with open(file_path, "r") as log_file:
        #     log_content = log_file.read()

    if save_csv == True:
        # Combine the data into a 2D array
        if len(frame_numbers) == len(float_values):
            data = list(zip(frame_numbers, float_values))
        else:
            frame_numbers.pop()
            data = list(zip(frame_numbers, float_values))

        # Save the extracted data to a new CSV file
        if save_csv == True:
            with open(output_file_name, "w", newline="") as output_csv:
                csv_writer = csv.writer(output_csv)
                csv_writer.writerow(["Frame Number", "Float Value"])
                csv_writer.writerows(data)
                print(f"Data extracted and saved to {output_file_name}")
    else:
        data = float_values

    return data


def load_fictrac_data_file(this_file, analysis_methods):
    # load analysis methods
    track_ball_radius = analysis_methods.get("trackball_radius")
    monitor_fps = analysis_methods.get("monitor_fps")
    camera_fps = analysis_methods.get("camera_fps")
    fictrac_posthoc_analysis = analysis_methods.get("fictrac_posthoc_analysis")
    overwrite_curated_dataset = analysis_methods.get("overwrite_curated_dataset")
    # laod file
    file_name = Path(this_file).stem
    experiment_timestamp = file_name.split("-")
    raw_data = pd.read_table(this_file, sep="\s+")
    """
    fictrac title is here 'frame counter', 'delta rotation vector cam x', 'delta rotation vector cam y', \
    'delta rotation vector cam z', 'delta rotation error score', 'delta rotation vector lab x', \
    'delta rotation vector lab y', 'delta rotation vector lab z', 'absolute rotation vector cam x',\
    'absolute rotation vector cam y', 'absolute rotation vector cam z', \
    'absolute rotation vector lab x', 'absolute rotation vector lab y', \
    'absolute rotation vector lab z', 'intergrated x position','intergrated y position', \
    'intergrated animal heading', 'animal movement direction', 'animal movement speed', \
    'intergrated forward motion','intergrated side motion', 'timestamp', 'sequence counter','delta timestamp', 'alt. timestamp']
    """
    ## drop some column
    raw_data.drop(
        raw_data.columns[
            [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23]
        ],
        axis=1,
        inplace=True,
    )
    raw_data.columns = [
        "delta rotation vector lab z",
        "intergrated x position",
        "intergrated y position",
        "integrated animal heading",
        "animal movement direction",
        "timestamp",
    ]
    ## cleaning , and change the data from object to float
    raw_data.iloc[:, :] = raw_data.iloc[:, :].replace({",": ""}, regex=True)
    if (
        raw_data["timestamp"][0] == raw_data["timestamp"][1]
        and fictrac_posthoc_analysis == False
    ):
        print(
            "this error timestamp value comes from a bug in new fictrac, extract timestamp from log file"
        )
        # f"{file_name}.log"
        # this_log_file = f"{file_name}.log"
        trial_event_pattern = "tracking*.csv"
        this_log_file = find_file(thisDir, trial_event_pattern)
        # timestamp_arr = generate_timestamp_csv(os.path.join(thisDir, this_log_file))
        timestamp_arr = generate_timestamp_csv(os.path.join(thisDir, this_log_file))
        ## needs to troubleshoot this part to make sure only the timestamp array is used to replace the original part
        if len(timestamp_arr) > raw_data.shape[0]:
            timestamp_arr.pop()
            print(
                "there is an additional one timestamp in the log file. Remove the last one"
            )

        elif len(timestamp_arr) < raw_data.shape[0]:
            fill_arr = np.arange(raw_data.shape[0] - len(timestamp_arr))
            test = np.hstack((fill_arr, timestamp_arr))
            timestamp_arr = test
            # timestamp_arr.append(timestamp_arr[-1])
            print(
                "missing timestamps from the bonsai log file. create some meaningless number to fill in that gap"
            )
        else:
            print("log file size matches with dat file")
        raw_data["timestamp"] = timestamp_arr
    elif fictrac_posthoc_analysis == True:
        print(
            "Use posthoc analysis to get dat file. Timestamps are locked to when the posthoc analysis is done. Hence, not useful to align tracking with stimulus anymore"
        )
    else:
        print("no error timestamp value comes from fictrac dat file")

    for ind in range(0, raw_data.shape[1] - 1):
        raw_data.iloc[:, ind] = pd.to_numeric(raw_data.iloc[:, ind]).astype("float32")
    ## adjust the unit of the x, y position from radiam to mm
    raw_data.loc[:, ["intergrated x position"]] = (
        raw_data.loc[:, ["intergrated x position"]] * track_ball_radius
    )
    raw_data.loc[:, ["intergrated y position"]] = (
        raw_data.loc[:, ["intergrated y position"]] * track_ball_radius
    )
    ## adjust the unit of the z vector based on the target frame rate of fictrac to get angular velocity omega
    raw_data.loc[:, ["delta rotation vector lab z"]] = (
        raw_data.loc[:, ["delta rotation vector lab z"]] * camera_fps
    )
    remove_old_fictrac_database = False
    if remove_old_fictrac_database & overwrite_curated_dataset:
        old_database_pattern = f"database_curated*.pickle"
        found_result = find_file(thisDir, old_database_pattern)
        if found_result.is_file() == True:
            try:
                Path.unlink(found_result)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
        elif isinstance(found_result, list):
            for this_file in found_result:
                try:
                    Path.unlink(this_file)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))

    ### save the curated_database
    if analysis_methods.get("save_output") == True:
        database_name = f"database_{file_name}.pickle"
        database_directory = this_file.parent.joinpath(database_name)
        if (overwrite_curated_dataset == False) and (
            database_directory.is_file() == True
        ):
            print(f"do not overwrite existing pickle file {this_file}")
        else:
            raw_data.to_pickle(database_directory)
        # old_database_name = f"database_curated.pickle"
        # old_database_dir = this_file.parent.joinpath(old_database_name)
        # if old_database_dir.is_file() == True:
        #     try:
        #         Path.unlink(old_database_dir)
        #     except OSError as e:
        #         print("Error: %s - %s." % (e.filename, e.strerror))


def preprocess_fictrac_data(thisDir, json_file):
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())
    # timestamp_pattern='trial_*.csv'
    log_pattern = "*.log"
    pd_pattern = "PD*.csv"
    # list up the files infind_file the dir

    dat_pattern = "*.dat"
    found_result = find_file(thisDir, dat_pattern)
    if found_result is None:
        return print(f"file with {dat_pattern} not found")
    else:
        if isinstance(found_result, list):
            for this_file in found_result:
                print(
                    f"found multiple files with {dat_pattern}. Use a for-loop to go through them"
                )
                load_fictrac_data_file(this_file, analysis_methods)
        elif len(found_result.stem) > 0:
            load_fictrac_data_file(found_result, analysis_methods)


if __name__ == "__main__":
    json_file = r"..\ephys\analysis_methods_dictionary.json"
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())
    #analysis_methods.update({"experiment_name":"looming"})
    analysis_methods.update({"experiment_name":"gratings"})
    #stim_directory=r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN25018\250519\looming\session1"
    stim_directory=r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN25018\250519\gratings\session1"
    trial_ext = "trial*.csv"
    this_csv = find_file(stim_directory, trial_ext)
    stim_pd = pd.read_csv(this_csv)
    meta_info, _ = sorting_trial_info(stim_pd,analysis_methods)
    #thisDir = r"Z:\DATA\experiment_trackball_Optomotor\MatrexVR\GN24036\240801\coherence\session1"
    thisDir = r"C:\Users\neuroLaptop\Documents\GN25006\250312\receding\session1"
    # thisDir = r"C:\Users\neuroLaptop\Documents\GN25040\250106\speed\session1"
    json_file = r".\analysis_methods_dictionary.json"
    tic = time.perf_counter()
    old_csv_path = r"Z:\DATA\experiment_trackball_Optomotor\MatrexVR\GN24124\241209\coherence\session1\lux4_2024-12-09T12_02_03.csv"
    new_csv_path = r"Z:\DATA\experiment_trackball_Optomotor\MatrexVR\GN24124\241209\coherence\session1\camera4_2024-12-09T12_02_03.csv"
    update_csv_value_pd(old_csv_path, new_csv_path, 0)
    # preprocess_fictrac_data(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
    # this_file = r"Z:\Users\chiyu\DL220THP_Thermo1_240904_240908.csv"
    # load_temperature_data(this_file)

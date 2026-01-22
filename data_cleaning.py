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
def remove_false_detection_heading(df, angle_col='heading_direction', threshold_upper=None, threshold_lower=None, threshold_range=None):
### written by Aljoscha Markus 27/06/2025
    y = df[angle_col].values
    dy = np.diff(y)

    # Identify initial flaw indices
    flaw = np.where((np.abs(dy) > threshold_lower) & (np.abs(dy) < threshold_upper))[0]

    if flaw.size == 0:
        pass
    else:
        # Initialize a mask for flawed values
        mask = np.zeros_like(y, dtype=bool)
        mask[flaw] = True  # mark the initial flaws

        # Expand the mask where flaws are close together
        d_flaw = np.diff(flaw)
        close_pairs = np.where(d_flaw < threshold_range)[0]

        for i in close_pairs:
            mask[flaw[i]:flaw[i + 1] + 1] = True

        # Apply mask and interpolate
        # y[mask] = np.nan
        # corrected_y=pd.Series(y).interpolate(method='linear').values
        # fig2, (ax1,ax2) = plt.subplots(
        # nrows=1, ncols=2, figsize=(27,10), tight_layout=True)
        # ax1.plot(np.arange(df[angle_col].values[200000:250000].shape[0]),df[angle_col].values[200000:250000])
        # ax2.plot(np.arange(corrected_y[200000:250000].shape[0]),corrected_y[200000:250000])
        # ax1.set(title="raw heading")
        # ax2.set(title="after remove_false_detection_heading")
        # fig2.savefig(f'heading_testing.png')


        df[angle_col] = pd.Series(y).interpolate(method='linear')
    return df


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
    x=None, y=None,distance=None,diskretise_length=24,
):  # discretize data into equidistant points, using body length (https://stackoverflow.com/questions/19117660/how-to-generate-equispaced-interpolating-values)
    # code writen by Sercan Sayin and described in (https://www.science.org/doi/10.1126/science.adq7832)
    # the source code can be found in (https://zenodo.org/records/14355590) # 12cm ,roughly 3BL
    if isinstance(x, list):
        analysis_len=len(x)    
    else:
        analysis_len=distance.shape[0]
    i, idx = 0, [0]
    while i < analysis_len - 1:
        cumulated_dist = 0
        for j in range(i + 1, analysis_len):
            if isinstance(x, list):
                cumulated_dist = math.sqrt((x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2)
            else:
                cumulated_dist=distance[j]-distance[i]
            if cumulated_dist > diskretise_length:
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
    stim_duration=analysis_methods.get("stim_duration")
    pre_stim_interval = analysis_methods.get("prestim_duration")
    stim_info = stim_info.reset_index()
    if type(stim_info.iloc[0,0])==str:
        first_event_time=stim_info.iloc[0,1]/camera_fps
    else:
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
    elif visual_paradigm_name.lower() in ["conflict","looming","receding","gratings","sweeping","flashing","choices"]:
        stim_variable = []
        stimulus_timestamp = []
        num_trial_col=np.shape(stim_info)[1] - 2
        for row in range(0, np.shape(stim_info)[0]):
            for col in range(2, np.shape(stim_info)[1]-1):
                if type(stim_info.iloc[row, col])==str:
                    if ')' in stim_info.iloc[row, col] and '}' in stim_info.iloc[row, col]:
                        continue
                    elif ')' in stim_info.iloc[row, col] and ';' in stim_info.iloc[row, col]:
                        continue##temperature and humidity can be processed with the following code
                    elif '}' in stim_info.iloc[row, col]:
                        stim_variable.append(
                            float(stim_info.iloc[row, col].split("=", 1)[1].split("}", 1)[0])
                        )
                    elif ')' in stim_info.iloc[row, col]:
                        stim_variable.append(
                            float(stim_info.iloc[row, col].split("=", 1)[1].split(")", 1)[0])
                        )
                    else:
                        stim_variable.append(
                            float(stim_info.iloc[row, col].split("=", 1)[1]))
                else:
                    continue ##time stamp can be processed with the following code
            if stim_info['Timestamp'].dtypes==object:
                timestamp = stim_info.loc[row, "Value"]
                stim_variable.append(float(timestamp.replace(")", "")))

        stim_variable = np.array(stim_variable).reshape((len(stim_info), -1))
        if visual_paradigm_name.lower() == "choices":
            default_column_names = ["LocationBeginX1","LocationEndX1","LocationBeginZ1","LocationEndZ1","PolarBeginR1","PolarEndR1","PolarBeginDegree1","PolarEndDegree1","Phase1","PreMovDuration","Duration","PostMovDuration","ISI","LocustObj1","LocustObj2","HeadingAt0degree1","HeadingAt0degree2","LocustTexture1","LocustTexture2","TranslationalGain","RotationalGain","R1","G1","B1","A1","R2","G2","B2","A2"] 
            cols_to_convert = ["LocustTexture1","LocustTexture2","HeadingAt0degree1","HeadingAt0degree2","LocustObj1","LocustObj2","PolarBeginDegree1","PolarEndDegree1","Phase1","Duration","TranslationalGain","RotationalGain"]
        else:
            default_column_names = ["LocationBeginX1","LocationEndX1","LocationBeginZ1","LocationEndZ1","PolarBeginR1","PolarEndR1","PolarBeginDegree1","PolarEndDegree1","Phase1","PreMovDuration","Duration","PostMovDuration","ISI","LocustObj1","ReverseZ1","LocustTexture1","TranslationalGain","RotationalGain","R1","G1","B1","A1","R2","G2","B2","A2"]
            cols_to_convert = ["LocustTexture1","ReverseZ1","LocustObj1","PolarBeginDegree1","PolarEndDegree1","Phase1","Duration"]
        if stim_info['Timestamp'].dtypes=='float64':
            this_column_names=default_column_names[:stim_variable.shape[1]]
            stim_info_curated = pd.DataFrame(stim_variable, columns=this_column_names)
            stim_info["ts"]=stim_info['Timestamp']
            stim_info['temperature']=pd.to_numeric(stim_info["Value"].str.replace(r"[()]", "", regex=True).str.split(";",expand=True)[0])## when there is a bug saying things can not be sliced, that usually means temperature is not logged in some particular column
            stim_info['humidity']=pd.to_numeric(stim_info["Value"].str.replace(r"[()]", "", regex=True).str.split(";",expand=True)[1])
            stim_info["temperature"] = stim_info["temperature"].astype("float32")
            stim_info["humidity"] = stim_info["humidity"].astype("float32")
            stim_info=pd.concat((stim_info_curated,stim_info.iloc[:,-3:]),axis=1)
        else:
            this_column_names=default_column_names[:stim_variable.shape[1]-1]
            this_column_names.append("ts")
            stim_info_curated = pd.DataFrame(stim_variable, columns=this_column_names)
            stim_info=stim_info_curated    
        
        stim_info[cols_to_convert] = stim_info[cols_to_convert].astype(int)
        begin_degree_sorted=sorted(stim_info["PolarBeginDegree1"].unique())

        if visual_paradigm_name.lower()=="choices":
            stim_type=['glocust_null','null_glocust','glocust_black','black_glocust','glocust_glocust','black_null','null_black','black_black']
            filters=[
            (stim_info["LocustTexture1"]==1)       
            &(stim_info["R1"] == 0)
            &(stim_info["G1"] == 0)
            &(stim_info["B1"] == 0)
            &(stim_info["A1"] == 1)
            &(stim_info["A2"] ==0),
            (stim_info["LocustTexture2"]==1)       
            &(stim_info["R2"] == 0)
            &(stim_info["G2"] == 0)
            &(stim_info["B2"] == 0)
            &(stim_info["A2"] == 1)
            &(stim_info["A1"] ==0),
            (stim_info["LocustTexture1"]==1)       
            &(stim_info["R1"] == 0)
            &(stim_info["G1"] == 0)
            &(stim_info["B1"] == 0)
            &(stim_info["A1"] == 1)
            &(stim_info["LocustTexture2"]==0)       
            &(stim_info["R2"] == 0)
            &(stim_info["G2"] == 0)
            &(stim_info["B2"] == 0)
            &(stim_info["A2"] ==1),
            (stim_info["LocustTexture1"]==0)       
            &(stim_info["R1"] == 0)
            &(stim_info["G1"] == 0)
            &(stim_info["B1"] == 0)
            &(stim_info["A1"] == 1)
            &(stim_info["LocustTexture2"]==1)       
            &(stim_info["R2"] == 0)
            &(stim_info["G2"] == 0)
            &(stim_info["B2"] == 0)
            &(stim_info["A2"] ==1),
            (stim_info["LocustTexture1"]==1)       
            &(stim_info["R1"] == 0)
            &(stim_info["G1"] == 0)
            &(stim_info["B1"] == 0)
            &(stim_info["A1"] == 1)
            &(stim_info["LocustTexture2"]==1)       
            &(stim_info["R2"] == 0)
            &(stim_info["G2"] == 0)
            &(stim_info["B2"] == 0)
            &(stim_info["A2"] ==1),
            (stim_info["LocustTexture1"]==0)       
            &(stim_info["R1"] == 0)
            &(stim_info["G1"] == 0)
            &(stim_info["B1"] == 0)
            &(stim_info["A1"] == 1)
            &(stim_info["A2"] ==0),
            (stim_info["LocustTexture2"]==0)       
            &(stim_info["R2"] == 0)
            &(stim_info["G2"] == 0)
            &(stim_info["B2"] == 0)
            &(stim_info["A2"] == 1)
            &(stim_info["A1"] ==0),
            (stim_info["LocustTexture1"]==0)       
            &(stim_info["R1"] == 0)
            &(stim_info["G1"] == 0)
            &(stim_info["B1"] == 0)
            &(stim_info["A1"] == 1)
            &(stim_info["LocustTexture2"]==0)       
            &(stim_info["R2"] == 0)
            &(stim_info["G2"] == 0)
            &(stim_info["B2"] == 0)
            &(stim_info["A2"] ==1)]
            stim_info["stim_type"] = np.select(filters, stim_type,default="unclassified") 
        elif visual_paradigm_name.lower() == "conflict":
            filters = [
            (stim_info["Phase1"] == 180)
            & (stim_info["PolarEndDegree1"] < stim_info["PolarBeginDegree1"]),
            (stim_info["Phase1"] == 0)
            & (stim_info["PolarEndDegree1"] < stim_info["PolarBeginDegree1"]),
            (stim_info["Phase1"] == 180)
            & (stim_info["PolarEndDegree1"] > stim_info["PolarBeginDegree1"]),
            (stim_info["Phase1"] == 0)
            & (stim_info["PolarEndDegree1"] > stim_info["PolarBeginDegree1"])]
            stim_type=["cc_back","cc_front","c_back","c_front"]
            stim_info["stim_type"] = np.select(filters, stim_type,default="unclassified")
        elif visual_paradigm_name.lower() == "gratings":
            stim_info["stim_type"] = stim_info["PolarBeginDegree1"].astype(int)
        elif visual_paradigm_name.lower()=="flashing":
            stim_type = ['black','blue','locust_yellow','white','green','locust_green']
            filters = [
            (stim_info["R1"] == 0)
            &(stim_info["G1"] == 0)
            &(stim_info["B1"] == 0),
            (stim_info["R1"] == 0)
            &(stim_info["G1"] == 0)
            &(stim_info["B1"] == 1),
            (stim_info["R1"] == 0.8117)
            (stim_info["R1"] == 1)
            &(stim_info["G1"] == 1)
            &(stim_info["B1"] == 1),
            (stim_info["R1"] == 0)
            &(stim_info["G1"] == 1)
            &(stim_info["B1"] == 0),
            (stim_info["R1"] == 0.5882)
        ]
            stim_info["stim_type"] = np.select(filters, stim_type,default="unclassified")
        elif 'R1' in this_column_names and visual_paradigm_name.lower()=="looming":
            stim_type = ['black','locust_green','locust_yellow','white','black_luminance','black_receding','grey','blue','blue_receding','blue_luminance','locust_yellow_receding','locust_yellow_luminance','white_receding','white_luminance']
            filters = [
            (stim_info["R1"] == 0)
            &(stim_info["G1"] == 0)
            &(stim_info["B1"] == 0)
            &(stim_info["A1"] == 1)
            &(stim_info["PolarBeginR1"] > stim_info["PolarEndR1"]),
            (stim_info["R1"] == 0.5882)
            &(stim_info["A1"] == 1)
            &(stim_info["PolarBeginR1"] > stim_info["PolarEndR1"]),
            (stim_info["R1"] == 0.8117)
            &(stim_info["A1"] == 1)
            &(stim_info["PolarBeginR1"] > stim_info["PolarEndR1"]),
            (stim_info["R1"] == 1)
            &(stim_info["G1"] == 1)
            &(stim_info["B1"] == 1)
            &(stim_info["A1"] == 1)
            &(stim_info["PolarBeginR1"] > stim_info["PolarEndR1"]),
            (stim_info["R1"] == 0)
            &(stim_info["G1"] == 0)
            &(stim_info["B1"] == 0)
            &(stim_info["A1"] == 0),
            (stim_info["R1"] == 0)
            &(stim_info["G1"] == 0)
            &(stim_info["B1"] == 0)
            &(stim_info["A1"] == 1)
            &(stim_info["PolarBeginR1"] < stim_info["PolarEndR1"]),
            (stim_info["R1"] == 0.6)
            &(stim_info["G1"] == 0.6)
            &(stim_info["B1"] == 0.6)
            &(stim_info["A1"] == 1)
            &(stim_info["PolarBeginR1"] > stim_info["PolarEndR1"]),
            (stim_info["R1"] == 0)
            &(stim_info["G1"] == 0)
            &(stim_info["B1"] == 1)
            &(stim_info["A1"] == 1)
            &(stim_info["PolarBeginR1"] > stim_info["PolarEndR1"]),
            (stim_info["R1"] == 0)
            &(stim_info["G1"] == 0)
            &(stim_info["B1"] == 1)
            &(stim_info["A1"] == 1)
            &(stim_info["PolarBeginR1"] < stim_info["PolarEndR1"]),
            (stim_info["R1"] == 0)
            &(stim_info["G1"] == 0)
            &(stim_info["B1"] == 1)
            &(stim_info["A1"] == 0),
            (stim_info["R1"] == 0.8117)
            &(stim_info["A1"] == 1)
            &(stim_info["PolarBeginR1"] < stim_info["PolarEndR1"]),
            (stim_info["R1"] == 0.8117)
            &(stim_info["A1"] == 0),              
            (stim_info["R1"] == 1)
            &(stim_info["G1"] == 1)
            &(stim_info["B1"] == 1)
            &(stim_info["A1"] == 1)
            &(stim_info["PolarBeginR1"] < stim_info["PolarEndR1"]),
            (stim_info["R1"] == 1)
            &(stim_info["G1"] == 1)
            &(stim_info["B1"] == 1)
            &(stim_info["A1"] == 0)
        ]
            stim_info["stim_type"] = np.select(filters, stim_type,default="unclassified")
        elif 'R2' in this_column_names and visual_paradigm_name.lower()=='sweeping':
            if stim_info['PolarBeginDegree1'].unique().shape[0]==2 or stim_info['PolarBeginR1'].unique().shape[0]==2:## need to add conditions here to include ccw cw horizontal
                if stim_info['PolarBeginDegree1'].unique().shape[0]==2:
                    degree1='PolarBeginDegree1'
                    degree2='PolarEndDegree1'
                else:
                    degree1='PolarBeginR1'
                    degree2='PolarEndR1'
                stim_type = ['black_ccw','black_cw','white_ccw','white_cw','yellow_ccw','yellow_cw']
                filters = [
                (stim_info["R1"] == 0)
                &(stim_info["G1"] == 0)
                &(stim_info["B1"] == 0)
                &(stim_info["A1"] == 1)
                &(stim_info[degree1]>stim_info[degree2]),
                (stim_info["R1"] == 0)
                &(stim_info["G1"] == 0)
                &(stim_info["B1"] == 0)
                &(stim_info["A1"] == 1)
                &(stim_info[degree1]<stim_info[degree2]),
                (stim_info["R1"] == 1)
                &(stim_info["G1"] == 1)
                &(stim_info["B1"] == 1)
                &(stim_info["A1"] == 1)
                &(stim_info[degree1]>stim_info[degree2]),
                (stim_info["R1"] == 1)
                &(stim_info["G1"] == 1)
                &(stim_info["B1"] == 1)
                &(stim_info["A1"] == 1)
                &(stim_info[degree1]<stim_info[degree2]),
                (stim_info["R1"] == 0.8117)
                &(stim_info[degree1]>stim_info[degree2]),
                (stim_info["R1"] == 0.8117)
                &(stim_info[degree1]<stim_info[degree2])]            
            elif stim_info['PolarBeginDegree1'].unique().shape[0]==1:
                stim_type = ['black_null','white_null','yellow_null','null_black','null_white','null_yellow','black_black','white_white','yellow_yellow','black_white','white_black','yellow_white','white_yellow','yellow_black','black_yellow']
                filters = [
                (stim_info["R1"] == 0)
                &(stim_info["G1"] == 0)
                &(stim_info["B1"] == 0)
                &(stim_info["A1"] == 1)
                &(stim_info["A2"] ==0),
                (stim_info["R1"] == 1)
                &(stim_info["G1"] == 1)
                &(stim_info["B1"] == 1)
                &(stim_info["A1"] == 1)
                &(stim_info["A2"] ==0),
                (stim_info["R1"] == 0.8117)
                &(stim_info["A1"] == 1)
                &(stim_info["A2"] ==0),
                (stim_info["R2"] == 0)
                &(stim_info["G2"] == 0)
                &(stim_info["B2"] == 0)
                &(stim_info["A2"] == 1)
                &(stim_info["A1"] ==0),
                (stim_info["R2"] == 1)
                &(stim_info["G2"] == 1)
                &(stim_info["B2"] == 1)
                &(stim_info["A2"] == 1)
                &(stim_info["A1"] ==0),
                (stim_info["R2"] == 0.8117)
                &(stim_info["A2"] == 1)
                &(stim_info["A1"] ==0),
                (stim_info["R1"] == 0)
                &(stim_info["G1"] == 0)
                &(stim_info["B1"] == 0)
                &(stim_info["A1"] == 1)
                &(stim_info["R2"] == 0)
                &(stim_info["G2"] == 0)
                &(stim_info["B2"] == 0)
                &(stim_info["A2"] == 1),
                (stim_info["R1"] == 1)
                &(stim_info["G1"] == 1)
                &(stim_info["B1"] == 1)
                &(stim_info["A1"] == 1)
                &(stim_info["R2"] == 1)
                &(stim_info["G2"] == 1)
                &(stim_info["B2"] == 1)
                &(stim_info["A2"] == 1),
                (stim_info["R1"] == 0.8117)
                &(stim_info["A1"] == 1)
                &(stim_info["R2"] == 0.8117)
                &(stim_info["A2"] == 1),
                (stim_info["R1"] == 0)
                &(stim_info["G1"] == 0)
                &(stim_info["B1"] == 0)
                &(stim_info["A1"] == 1)
                &(stim_info["R2"] == 1)
                &(stim_info["G2"] == 1)
                &(stim_info["B2"] == 1)
                &(stim_info["A2"] == 1),
                (stim_info["R1"] == 1)
                &(stim_info["G1"] == 1)
                &(stim_info["B1"] == 1)
                &(stim_info["A1"] == 1)
                &(stim_info["R2"] == 0)
                &(stim_info["G2"] == 0)
                &(stim_info["B2"] == 0)
                &(stim_info["A2"] == 1),
                (stim_info["R1"] == 0.8117)
                &(stim_info["A1"] == 1)
                &(stim_info["R2"] == 1)
                &(stim_info["G2"] == 1)
                &(stim_info["B2"] == 1)
                &(stim_info["A2"] == 1),
                (stim_info["R2"] == 0.8117)
                &(stim_info["A2"] == 1)
                &(stim_info["R1"] == 1)
                &(stim_info["G1"] == 1)
                &(stim_info["B1"] == 1)
                &(stim_info["A1"] == 1),
                (stim_info["R1"] == 0.8117)
                &(stim_info["A1"] == 1)
                &(stim_info["R2"] == 0)
                &(stim_info["G2"] == 0)
                &(stim_info["B2"] == 0)
                &(stim_info["A2"] == 1),
                (stim_info["R2"] == 0.8117)
                &(stim_info["A2"] == 1)
                &(stim_info["R1"] == 0)
                &(stim_info["G1"] == 0)
                &(stim_info["B1"] == 0)
                &(stim_info["A1"] == 1)
                ]
            stim_info["stim_type"] = np.select(filters, stim_type,default="unclassified")
        elif 'R1' in this_column_names and visual_paradigm_name.lower()== "sweeping":
            stim_type = ['black_dir1','locust_green_dir1','locust_yellow_dir1','white_dir1','black_di2','locust_green_dir2','locust_yellow_dir2','white_dir2','green_dir1','green_dir2']#dir2 means downward; dir1 means upward
            filters = [
            (stim_info["R1"] == 0)
            &(stim_info["G1"] == 0)
            &(stim_info["B1"] == 0)
            &(stim_info["A1"] == 1)
            &(stim_info["PolarBeginDegree1"] > stim_info["PolarEndDegree1"]),
            (stim_info["R1"] == 0.5882)
            &(stim_info["A1"] == 1)
            &(stim_info["PolarBeginDegree1"] > stim_info["PolarEndDegree1"]),
            (stim_info["R1"] == 0.8117)
            &(stim_info["A1"] == 1)
            &(stim_info["PolarBeginDegree1"] > stim_info["PolarEndDegree1"]),
            (stim_info["R1"] == 1)
            &(stim_info["G1"] == 1)
            &(stim_info["B1"] == 1)
            &(stim_info["A1"] == 1)
            &(stim_info["PolarBeginDegree1"] > stim_info["PolarEndDegree1"]),
            (stim_info["R1"] == 0)
            &(stim_info["G1"] == 0)
            &(stim_info["B1"] == 0)
            &(stim_info["A1"] == 1)
            &(stim_info["PolarBeginDegree1"] < stim_info["PolarEndDegree1"]),
            (stim_info["R1"] == 0.5882)
            &(stim_info["A1"] == 1)
            &(stim_info["PolarBeginDegree1"] < stim_info["PolarEndDegree1"]),
            (stim_info["R1"] == 0.8117)
            &(stim_info["A1"] == 1)
            &(stim_info["PolarBeginDegree1"] < stim_info["PolarEndDegree1"]),
            (stim_info["R1"] == 1)
            &(stim_info["G1"] == 1)
            &(stim_info["B1"] == 1)
            &(stim_info["A1"] == 1)
            &(stim_info["PolarBeginDegree1"] < stim_info["PolarEndDegree1"]),
            (stim_info["R1"] == 0)
            &(stim_info["G1"] == 1)
            &(stim_info["B1"] == 0)
            &(stim_info["A1"] == 1)
            &(stim_info["PolarBeginDegree1"] > stim_info["PolarEndDegree1"]),
            (stim_info["R1"] == 0)
            &(stim_info["G1"] == 1)
            &(stim_info["B1"] == 0)
            &(stim_info["A1"] == 1)
            &(stim_info["PolarBeginDegree1"] < stim_info["PolarEndDegree1"]),
        ]
            stim_info["stim_type"] = np.select(filters, stim_type,default="unclassified")
        else:
            stim_type = [
                "receding_left",
                "receding_center",
                "receding_right",
                "looming_left",
                "looming_center",
                "looming_right",
            ]
            filters = [
            (stim_info["PolarBeginR1"] < stim_info["PolarEndR1"])
            &(stim_info["PolarEndDegree1"] == begin_degree_sorted[2]),
            (stim_info["PolarBeginR1"] < stim_info["PolarEndR1"])
            &(stim_info["PolarEndDegree1"] == begin_degree_sorted[1]),
            (stim_info["PolarBeginR1"] < stim_info["PolarEndR1"])
            &(stim_info["PolarEndDegree1"] == begin_degree_sorted[0]),
            (stim_info["PolarBeginR1"] > stim_info["PolarEndR1"])
            &(stim_info["PolarEndDegree1"] == begin_degree_sorted[2]),
            (stim_info["PolarBeginR1"] > stim_info["PolarEndR1"])
            &(stim_info["PolarEndDegree1"] == begin_degree_sorted[1]),
            (stim_info["PolarBeginR1"] > stim_info["PolarEndR1"])
            &(stim_info["PolarEndDegree1"] == begin_degree_sorted[0])]
            stim_info["stim_type"] = np.select(filters, stim_type,default="unclassified")
        ## update the stim_type if the locustTexture1 is 1    
        if stim_info['LocustTexture1'].max()==1 and visual_paradigm_name=="looming":
            stim_info["stim_type"][stim_info['LocustTexture1']==1]='gregarious_locust'
        if visual_paradigm_name == "sweeping":### these additional conditions is needed in bilateral sequence assay because in the protocol, each stimulus is 2 sec but repeats multiple rounds
            if type(stim_duration)==list:
                if len(stim_duration)==1:
                    stim_info["Duration"] = np.repeat(analysis_methods.get("stim_duration")[0], stim_info.shape[0])
            elif type(stim_duration)==int or type(stim_duration)==float:
                stim_info["Duration"] = np.repeat(analysis_methods.get("stim_duration"), stim_info.shape[0])
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
            if stim_info['Timestamp'].dtypes=='float64':
                stim_info["ts"]=stim_info['Timestamp']
                stim_info['temperature']=pd.to_numeric(stim_info["Value"].str.replace(r"[()]", "", regex=True).str.split(";",expand=True)[0])
                stim_info['humidity']=pd.to_numeric(stim_info["Value"].str.replace(r"[()]", "", regex=True).str.split(";",expand=True)[1])
                stim_info["temperature"] = stim_info["temperature"].astype("float32")
                stim_info["humidity"] = stim_info["humidity"].astype("float32")
            else:
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


def load_fictrac_data_file(this_file, analysis_methods,column_to_drop=[0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23]):
    # load analysis methods
    
    track_ball_radius = analysis_methods.get("trackball_radius",5)## track ball radius in (cm)
    monitor_fps = analysis_methods.get("monitor_fps")
    camera_fps = analysis_methods.get("camera_fps")
    fictrac_posthoc_analysis = analysis_methods.get("fictrac_posthoc_analysis")
    overwrite_curated_dataset = analysis_methods.get("overwrite_curated_dataset")
    # laod file
    file_name = Path(this_file).stem
    thisDir=Path(this_file).parent
    experiment_timestamp = file_name.split("-")
    raw_data = pd.read_table(this_file, sep="\s+")
    # 
    # column7=raw_data.iloc[:,7].astype(float)
    # column6=raw_data.iloc[:,6].astype(float)
    # column5=raw_data.iloc[:,5].astype(float)
    # print(np.sum(abs(column5.values)),np.sum(abs(column6.values)),np.sum(abs(column7.values)))
    # fig1, (ax, ax1,ax2) = plt.subplots(
    #     nrows=3, ncols=1, figsize=(18, 7), tight_layout=True
    # )
    # ax.plot(column5.values)
    # ax1.plot(column6.values)
    # ax2.plot(column7.values)
    # plt.show()
    # fig1.savefig(f"{this_file}_raw_fictrac_check.png")
    """
    fictrac title is here 'frame counter', 'delta rotation vector cam x', 'delta rotation vector cam y', \
    'delta rotation vector cam z', 'delta rotation error score', 'delta rotation vector lab x', \
    'delta rotation vector lab y', 'delta rotation vector lab z', 'absolute rotation vector cam x',\
    'absolute rotation vector cam y', 'absolute rotation vector cam z', \
    'absolute rotation vector lab x', 'absolute rotation vector lab y', \
    'absolute rotation vector lab z', 'intergrated x position','intergrated y position', \
    'intergrated animal heading', 'animal movement direction', 'animal movement speed', \
    'intergrated forward motion','intergrated side motion', 'timestamp', 'sequence counter','delta timestamp', 'alt. timestamp']
    In previous Z ball settings, until 20204-07-01, Z axis is the yaw axis, X is the roll axis, Y is the pitch axis. positive X is the anterior, ventral is the positive Z, lateral is the positive Y
    In the latest Z ball settings, from 2025-09-01, Z axis is the roll axis, X is the yaw axis, Y is the pitch axis. positive X is the anterior, dorsal is the positive Z, lateral is the positive Y.
    Also in the latest Z ball settings, from 2025-09-01, campera fps is 100Hz
    """
    ## drop some column
    raw_data.drop(
        raw_data.columns[column_to_drop],
        axis=1,
        inplace=True,
    )
    raw_data.columns = [
        "delta rotation vector lab x",
        "delta rotation vector lab y",
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
    ## adjust the unit of the x, y position from radiam to cm
    raw_data[["intergrated x position","intergrated y position"]]=raw_data[["intergrated x position","intergrated y position"]]*track_ball_radius
    ## adjust the unit of the z vector based on the target frame rate of fictrac to get angular velocity omega
    raw_data[["delta rotation vector lab x","delta rotation vector lab y","delta rotation vector lab z"]]=raw_data[["delta rotation vector lab x","delta rotation vector lab y","delta rotation vector lab z"]]*camera_fps
    remove_old_fictrac_database = True
    if remove_old_fictrac_database & overwrite_curated_dataset:
        old_database_pattern = f"database_curated*.pickle"
        found_result = find_file(thisDir, old_database_pattern)
        if found_result is None:
            print("did not find any pickle curated database")
            pass
        elif isinstance(found_result, list):
            for this_file in found_result:
                try:
                    Path.unlink(this_file)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))
        elif found_result.is_file() == True:
            try:
                Path.unlink(found_result)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))


    ### save the curated_database
    if analysis_methods.get("save_output") == True:
        parquet_name = f"database_{file_name}.parquet.gzip"
        parquet_directory = this_file.parent.joinpath(parquet_name)
        if (overwrite_curated_dataset == False) and (
            parquet_directory.is_file() == True
        ):
            print(f"do not overwrite existing pickle file {this_file}")
        else:
            #raw_data.to_pickle(database_directory)
            raw_data.to_parquet(parquet_directory,compression='gzip')
    return raw_data


def preprocess_fictrac_data(thisDir, json_file,column_to_drop=[0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23]):
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
                _=load_fictrac_data_file(this_file, analysis_methods,column_to_drop)
        elif len(found_result.stem) > 0:
            _=load_fictrac_data_file(found_result, analysis_methods,column_to_drop)


if __name__ == "__main__":
    json_file = r"..\ephys\analysis_methods_dictionary.json"
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())
    #analysis_methods.update({"experiment_name":"looming"})
    #analysis_methods.update({"experiment_name":"gratings"})
    #analysis_methods.update({"experiment_name":"coherence"})
    #stim_directory=r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN25018\250519\looming\session1"
    # stim_directory=r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN25018\250519\gratings\session1"
    # trial_ext = "trial*.csv"
    # this_csv = find_file(stim_directory, trial_ext)
    # stim_pd = pd.read_csv(this_csv)
    # meta_info, _ = sorting_trial_info(stim_pd,analysis_methods)
    #thisDir = r"Z:\DATA\experiment_trackball_Optomotor\MatrexVR\GN24036\240801\coherence\session1"
    #thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN25102\250929\coherence\session1"
    #thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN25101\250917\looming\session1"
    thisDir = r"Y:\GN25051\251101\sweeping\session1"
    #thisDir = r"C:\Users\neuroLaptop\Documents\GN25006\250312\receding\session1"
    # thisDir = r"C:\Users\neuroLaptop\Documents\GN25040\250106\speed\session1"
    json_file = r"..\ephys\analysis_methods_dictionary.json"
    tic = time.perf_counter()
    # old_csv_path = r"Z:\DATA\experiment_trackball_Optomotor\MatrexVR\GN24124\241209\coherence\session1\lux4_2024-12-09T12_02_03.csv"
    # new_csv_path = r"Z:\DATA\experiment_trackball_Optomotor\MatrexVR\GN24124\241209\coherence\session1\camera4_2024-12-09T12_02_03.csv"
    # update_csv_value_pd(old_csv_path, new_csv_path, 0)
    preprocess_fictrac_data(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
    # this_file = r"Z:\Users\chiyu\DL220THP_Thermo1_240904_240908.csv"
    # load_temperature_data(this_file)

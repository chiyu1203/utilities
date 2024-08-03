import pandas as pd
import numpy as np
import os
import re
import csv
import json
import time

# import chardet
import sys
from pathlib import Path
from useful_tools import find_file


def update_csv_value(old_csv_path, new_csv_path):
    with open(old_csv_path, "r") as x:
        old_csv = list(csv.reader(x, delimiter=","))
        # old_csv = np.genfromtxt(x,delimiter=",")
    old_csv = np.genfromtxt(old_csv_path, delimiter=",")
    new_csv = np.genfromtxt(new_csv_path, delimiter=",", dtype=int)
    old_csv[:, 0] = new_csv
    np.savetxt(old_csv_path, old_csv, delimiter=",", fmt="%1.4f")
    return None


def load_temperature_data(txt_path):
    if txt_path.endswith(".txt"):
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

    elif txt_path.endswith(".csv"):
        print("Here to process data loaded in Bonsai")

    return df


def sorting_trial_info(stim_info, visual_paradigm_name="coherence", exp_place="Zball"):
    stim_info = stim_info.reset_index()
    raw_column_number = stim_info.shape[1]
    if visual_paradigm_name == "gratings":
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
        stim_type = np.sort(stim_info.stim_type.unique()).tolist()

    elif visual_paradigm_name == "sweeploom":
        stim_variable = []
        stimulus_timestamp = []
        stim_type = analysis_methods.get("stim_type")
        # stim_type = [
        #     "c2b_f_sloom",
        #     "c2b_s_sloom",
        #     "c2f_f_sloom",
        #     "c2f_s_sloom",
        #     "c2c_f_loom",
        #     "c2c_s_loom",
        # ]
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
    else:

        col_index = [2, 4, 5, 7, 11, 13, 15, 17]
        col_name = [
            "ID",
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
        if exp_place == "Zball" or exp_place == "matrexVR":
            stim_info["ts"] = stim_info["Value"].str.replace(r"[()]", "", regex=True)
        elif exp_place == "VCCball":
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
                stim_info["VelX"] * stim_info["numDots"] / abs(stim_info["VelX"])
            )
        elif visual_paradigm_name == "coherence":
            stim_variable_direction = (
                stim_info["VelX"] * stim_info["Coherence"] / abs(stim_info["VelX"])
            )
        # positive value means dots moves from right to left in a window, allocentric orientation in a panoramic setup: counterclock wise
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
    fictrac_posthoc_analysis = analysis_methods.get("fictrac_posthoc_analysis")
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
        raw_data.columns[[0, 1, 2, 3, 4, 8, 9, 10, 16, 17, 19, 20, 21, 22, 23]],
        axis=1,
        inplace=True,
    )
    raw_data.columns = [
        "delta rotation vector lab x",
        "delta rotation vector lab y",
        "delta rotation vector lab z",
        "absolute rotation vector lab x",
        "absolute rotation vector lab y",
        "absolute rotation vector lab z",
        "intergrated x position",
        "intergrated y position",
        "instant speed",
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
            "Use posthoc analysis to get dat file hence the timestamp from this file the time posthoc analysis was done. Hence, not useful to align tracking with stimulus anymore"
        )
    else:
        print("no error timestamp value comes from fictrac dat file")
    for ind in range(0, raw_data.shape[1] - 1):
        raw_data.iloc[:, ind] = pd.to_numeric(raw_data.iloc[:, ind]).astype("float32")

    ## adjust the unit of the x, y position
    raw_data.iloc[:, 6] = raw_data.iloc[:, 6] * track_ball_radius
    raw_data.iloc[:, 7] = raw_data.iloc[:, 7] * track_ball_radius
    raw_data.iloc[:, 8] = raw_data.iloc[:, 8] * track_ball_radius * monitor_fps
    ## adjust the unit of the z vector based on the target frame rate of fictrac
    raw_data.iloc[:, 2] = raw_data.iloc[:, 2] * monitor_fps

    ### save the curated_database
    if analysis_methods.get("debug_mode") == False:
        database_name = f"database_{file_name}.pickle"
        database_directory = this_file.parent.joinpath(database_name)
        if (analysis_methods.get("overwrite_curated_dataset") == False) and (
            database_directory.is_file() == True
        ):
            print(f"do not overwrite existing pickle file {this_file}")
        else:
            raw_data.to_pickle(database_directory)


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
    thisDir = r"Z:\DATA\experiment_trackball_Optomotor\MatrexVR\GN24032\240717\coherence\session1"
    json_file = r".\analysis_methods_dictionary.json"
    tic = time.perf_counter()
    preprocess_fictrac_data(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")

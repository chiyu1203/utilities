import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import time
import fnmatch
from pathlib import Path
import glob
import scipy.fftpack
import scipy.io
import pickle
import pandas as pd


def mat_converter(file):
    if type(file) == str:
        file = Path(file)
    ##this function converted pandas dataframe into mat file
    this_dir = file.parent
    with open(file, "rb") as f:
        tmp = pickle.load(f)
    if isinstance(tmp, pd.DataFrame):
        scipy.io.savemat(
            this_dir / "behavioural_summary.mat",
            mdict={"behavioural_summary": tmp.values},
        )
    elif isinstance(tmp, np.array):
        scipy.io.savemat(
            this_dir / "behavioural_summary.mat", mdict={"behavioural_summary": tmp}
        )


def calculate_fft(arr, sample_rate):
    # x = np.linspace(0.0, N*T, N)
    N = arr.size
    yf = scipy.fftpack.fft(arr)
    # xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    xf = scipy.fftpack.fftfreq(N, d=1 / sample_rate)[: N // 2]

    plt.plot(xf, 2.0 / N * np.abs(yf[0 : N // 2]))
    plt.grid()
    plt.show()
    return yf, xf


def find_nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


def select_animals(df, *args):
    if (len(args) % 2) == 1:
        print("ERROR input argument is odd number")
        return df
    elif len(args) == 0:
        print("ERROR no input argument")
        return df
    elif len(args) == 2:
        animal_of_interest = df[(df[args[0]] == args[1])]
        return animal_of_interest
    elif len(args) == 4:
        animal_of_interest = df[(df[args[0]] == args[1]) & (df[args[2]] == args[3])]
        return animal_of_interest
    elif len(args) == 6:
        animal_of_interest = df[
            (df[args[0]] == args[1])
            & (df[args[2]] == args[3])
            & (df[args[4]] == args[5])
        ]
        return animal_of_interest
    elif len(args) == 8:
        animal_of_interest = df[
            (df[args[0]] == args[1])
            & (df[args[2]] == args[3])
            & (df[args[4]] == args[5])
            & (df[args[6]] == args[7])
        ]
        return animal_of_interest
    else:
        print(
            "WARNING this looks like a lot of conditions....Are you sure what you are doing. Return df"
        )
        return df


def select_animals_gpt(df, *args):
    # Check if the number of input arguments is odd or zero.
    if len(args) % 2 == 1 or len(args) == 0:
        print("ERROR: Invalid number of input arguments")
        return df

    # Initialize a boolean mask with True values.
    mask = df.apply(lambda x: True, axis=1)

    for i in range(0, len(args), 2):
        column_name = args[i]
        value = args[i + 1]

        # Update the mask based on the current condition.
        mask &= df[column_name] == value

    # Apply the final mask to the DataFrame to select the animals of interest.
    animal_of_interest = df[mask]
    return animal_of_interest


def plot_single_variables(ax, variable_value, color, y_title, yaxis_range=None):
    ax.scatter(np.ones(len(variable_value)), variable_value, c=color, s=10, cmap="gray")
    ax.set_ylabel(y_title)
    if yaxis_range is not None:
        ax.set_ylim(yaxis_range)


def plot_multiple_variables(
    ax,
    index,
    variable_value,
    color,
    x_title,
    y_title,
    xaxis_range=None,
    yaxis_range=None,
):
    ax.scatter(index, variable_value, c=color, s=10, cmap="gray")
    ax.set_ylabel(y_title)
    ax.set_xlabel(x_title)
    if xaxis_range is not None:
        ax.set_xlim(xaxis_range)
    if yaxis_range is not None:
        ax.set_ylim(yaxis_range)


def find_file(thisDir, pattern1, pattern2=None, include_matching_both=True):
    all_files = os.listdir(thisDir)
    matched_pattern1 = fnmatch.filter(all_files, pattern1)

    # Find files that match pattern2
    if pattern2 != None:
        matched_pattern2 = fnmatch.filter(all_files, pattern2)
        if include_matching_both:
            file_check = list(set(matched_pattern1).union(matched_pattern2))
        else:

            # Avoid files that match both patterns by using set difference
            only_pattern1 = set(matched_pattern1) - set(matched_pattern2)
            only_pattern2 = set(matched_pattern2) - set(matched_pattern1)

            # Combine both sets to get final result
            file_check = list(only_pattern1.union(only_pattern2))
    else:
        file_check = matched_pattern1

    if len(file_check) == 0:
        print(f"no pattern found in {thisDir}. Let's leave this programme")
        return None
    elif len(file_check) == 1:
        return Path(thisDir) / file_check[0]
    else:
        vid_list = []
        for i in range(len(file_check)):
            vid_list.append(Path(thisDir) / file_check[i])
        return vid_list


def find_file_multiple_patterns(thisDir, patterns):
    if isinstance(thisDir, str):
        for pattern in patterns:
            led_files = glob.glob(os.path.join(thisDir, pattern))
            if led_files:
                return led_files
        print("led file is not found")
    else:
        for pattern in patterns:
            led_files = glob.glob(str(thisDir / pattern))
            if led_files:
                return led_files
        print("led file is not found")
    return None


def find_subdirectory(directory, exp_name, pattern):
    matching_subdirectory = []
    for root, dirs, files in os.walk(directory):
        if exp_name in root.split(
            os.path.sep
        ):  # Check for exp_name in the directory path
            for name in files:
                if pattern in name:
                    matching_subdirectory.append(root.replace("\\", "/"))
                    break  # No need to search further in this directory


def find_subdirectories_with_GPT(directory, pattern):
    matching_subdirectories = []

    for root, subdirs, files in os.walk(directory):
        for filename in files:
            if pattern in filename:
                matching_subdirectories.append(root)
                break  # No need to search further in this directory

    return matching_subdirectories


def hdf_file_reclocation(source_path, folder_name, overwrite_existing_tracking):
    source_file_list = os.listdir(source_path)
    sleap_file_pattern = "*analysis.h5"
    Datasets = "Z:/DATA/experiment_Locust_PreferenceArena"
    thisDataset = f"{Datasets}/{folder_name}"

    for source_file in source_file_list:
        # Extract the video ID from the data file name
        file_path = os.path.join(source_path, source_file)
        if "video20" in source_file:
            vidID = source_file.split("video20")[1].split(".analysis")[0]
        else:
            continue

        matching_subdirectories = find_subdirectories_with_GPT(thisDataset, vidID)
        if matching_subdirectories:
            subdirectory = matching_subdirectories[0]
            file_check = fnmatch.filter(os.listdir(subdirectory), sleap_file_pattern)
            if len(file_check) == 0:
                try:
                    shutil.move(file_path, subdirectory)
                    print(f"Moved {source_file} to {subdirectory}")
                except shutil.Error as e:
                    print(f"Error: {e}")
                continue
            elif overwrite_existing_tracking == True:
                shutil.copy2(file_path, subdirectory)
                os.remove(file_path)
                if source_file not in file_check:
                    for i in range(len(file_check)):
                        Path(os.path.join(subdirectory, file_check[i])).unlink()
                print(f"Overwrite the tracking file in {subdirectory} ")
            else:
                try:
                    shutil.move(file_path, subdirectory)
                except shutil.Error as e:
                    print(f"DONT OVERWRITE TRACKING FILE")
                    print(f"Error: {e}")
                continue
        else:
            print("No matching subdirectories found.")
            continue


# if __name__ == "__main__":
#     source_path = "Z:/DATA/experiment_Locust_PreferenceArena/sleap_hdf_new"
#     overwrite_existing_tracking = True
#     folder_name = "gregarisation"
#     hdf_file_reclocation(source_path, folder_name, overwrite_existing_tracking)

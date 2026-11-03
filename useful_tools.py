import matplotlib.pyplot as plt
import numpy as np
import os,json,glob,pickle,fnmatch,shutil
from pathlib import Path
import scipy.fftpack
import scipy.io
import pandas as pd
import scipy.stats as st
from typing import Optional, Sequence
import warnings,argparse
def read_seq_config(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Extract the list of sequences
    sequences = data['sequences']

    # Create the DataFrame with the desired columns
    df = pd.DataFrame([
        {
            'sceneName': item.get('sceneName'),
            'Duration': item.get('duration'),
            'configFile': item.get('parameters', {}).get('configFile')
        }
        for item in sequences
    ])
    return df

# https://stackoverflow.com/questions/40642061/how-to-set-axis-ticks-in-multiples-of-pi-python-matplotlib
def multiple_formatter(denominator=2, number=np.pi, latex="\pi"):
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def _multiple_formatter(x, pos):
        den = denominator
        num = int(np.rint(den * x / number))
        com = gcd(num, den)
        (num, den) = (int(num / com), int(den / com))
        if den == 1:
            if num == 0:
                return r"$0$"
            if num == 1:
                return r"$%s$" % latex
            elif num == -1:
                return r"$-%s$" % latex
            else:
                return r"$%s%s$" % (num, latex)
        else:
            if num == 1:
                return r"$\frac{%s}{%s}$" % (latex, den)
            elif num == -1:
                return r"$\frac{-%s}{%s}$" % (latex, den)
            else:
                return r"$\frac{%s%s}{%s}$" % (num, latex, den)

    return _multiple_formatter


class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex="\pi"):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(
            multiple_formatter(self.denominator, self.number, self.latex)
        )
def get_fill_between_range(data,confidence_interval=True,circular_statistics=False):
    if circular_statistics:
        mean_data=st.circmean(data,high=180,low=-180,axis=0)
        if confidence_interval:
            pass
        else:
            #sem_response = st.circstd(data, axis=0, ddof=1) / np.sqrt(data.shape[0])
            std_response = st.circstd(data,high=180,low=-180,axis=0)
            dif_y1=mean_data + std_response
            dif_y2=mean_data - std_response        
    else:
        mean_data=np.nanmean(data,axis=0)
        sem_response = np.nanstd(data, axis=0, ddof=1) / np.sqrt(data.shape[0])
        if confidence_interval:
            ##to plot distribution with 95% confidence interval with t distribution (since the sample is usually not big)
            confidence_level = 0.95
            
            sem_response = np.nanstd(data, axis=0, ddof=1) / np.sqrt(data.shape[0])
            cl95=st.t.interval(confidence=confidence_level, df=len(data)-1, loc=mean_data, scale=sem_response)
            #cl95=st.norm.interval(confidence_level,loc=mean_data,scale=st.sem(data))
            dif_y1=cl95[0][:]
            dif_y2=cl95[1][:]
            # a = 1.0 * np.array(data)
            # n = len(a)
            # m, se = np.mean(a), scipy.stats.sem(a)
            # h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        else:
            dif_y1=mean_data + sem_response
            dif_y2=mean_data - sem_response
    return dif_y1,dif_y2
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

def column_name_list(number,name):
    #genearate a list of column names with a given number and name
    #for example: number=5, name="animal" will generate ["animal0","animal1","animal2","animal3","animal4"]
    c_name_list=[]
    for i in range(number):
        c_name_list.append(f"{name}{i}")
    return c_name_list

def calculate_fft(arr, sample_rate):
    plot_result = False
    # x = np.linspace(0.0, N*T, N)
    N = arr.size
    yf = scipy.fftpack.fft(arr)
    # xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    xf = scipy.fftpack.fftfreq(N, d=1 / sample_rate)[: N // 2]
    if plot_result:
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

def boolean_array_to_event_table(
    reached_target: np.ndarray,
    *,
    timepoints: Optional[Sequence[float]] = None,
    sampling_interval: Optional[float] = None,
    stimulus_onset: float = 0.0,
    stimulus_onset_index: int = -1,
    consecutive_samples: int = 1,
    trial_ids: Optional[Sequence] = None,
    subject_ids: Optional[Sequence] = None,
    trial_col: str = "trial",
    subject_col: str = "animal_id",
) -> pd.DataFrame:
    """Convert a trial-by-time Boolean array to first-passage event data.

    Parameters
    ----------
    reached_target
        Two-dimensional array with shape (n_trials, n_timepoints). True means
        that the target-speed criterion was met at that sampled timepoint.
        Numeric arrays are accepted only when every value is 0 or 1.
    timepoints
        Optional length-n_timepoints vector containing the recorded times in
        seconds. Values may include baseline times. ``stimulus_onset`` is
        subtracted from this vector, and only strictly post-onset samples are
        used. Supply either ``timepoints`` or ``sampling_interval``.
    sampling_interval
        Seconds between columns when an explicit time vector is unavailable.
        Relative time is calculated as
        ``(column_index - stimulus_onset_index) * sampling_interval``.
    stimulus_onset
        Stimulus-onset time in the coordinate system of ``timepoints``. This
        argument is used only when ``timepoints`` is supplied.
    stimulus_onset_index
        Column representing t=0 when ``sampling_interval`` is used. The default
        -1 means that column 0 is the first post-onset sample, occurring one
        sampling interval after onset. If column 0 is sampled exactly at t=0,
        pass 0. If column 100 is t=0, pass 100.
    consecutive_samples
        Number of consecutive True samples required to define a hit. The event
        time is the beginning of the first qualifying run. Default is 1.
    trial_ids, subject_ids
        Optional identifiers with one value per row. Supplying ``subject_ids``
        adds a column suitable for ``subject_col`` in the fitting function.

    Returns
    -------
    pandas.DataFrame
        One row per trial. ``time`` is the first-hit latency for event=1, or
        the last observed post-onset time for event=0 (right-censored).

    Notes
    -----
    A sample exactly at t=0 is excluded because the continuous Weibull model
    requires positive event times. A True value at t=0 that persists into the
    next sample is therefore recorded at the first strictly positive time.
    """
    values = np.asarray(reached_target)
    if values.ndim != 2:
        raise ValueError(
            "reached_target must be a 2D array with shape "
            "(n_trials, n_timepoints)"
        )
    n_trials, n_timepoints = values.shape
    if n_trials == 0 or n_timepoints == 0:
        raise ValueError("reached_target cannot have an empty dimension")

    if values.dtype != np.bool_:
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("reached_target must contain Boolean or 0/1 values")
        if np.any(~np.isfinite(values)) or not np.all(np.isin(values, [0, 1])):
            raise ValueError("Numeric reached_target arrays may contain only 0 and 1")
    hits = values.astype(bool, copy=False)

    if timepoints is not None and sampling_interval is not None:
        raise ValueError("Supply timepoints or sampling_interval, not both")
    if timepoints is None and sampling_interval is None:
        raise ValueError("Supply either timepoints or sampling_interval")

    if timepoints is not None:
        sample_times = np.asarray(timepoints, dtype=float)
        if sample_times.ndim != 1 or len(sample_times) != n_timepoints:
            raise ValueError(
                "timepoints must be one-dimensional with one value per column"
            )
        if np.any(~np.isfinite(sample_times)):
            raise ValueError("timepoints contains missing or non-finite values")
        if np.any(np.diff(sample_times) <= 0):
            raise ValueError("timepoints must be strictly increasing")
        relative_times = sample_times - float(stimulus_onset)
    else:
        sampling_interval = float(sampling_interval)
        if not np.isfinite(sampling_interval) or sampling_interval <= 0:
            raise ValueError("sampling_interval must be a positive number")
        if not isinstance(stimulus_onset_index, (int, np.integer)):
            raise TypeError("stimulus_onset_index must be an integer")
        if stimulus_onset_index >= n_timepoints:
            raise ValueError("stimulus_onset_index lies beyond the final column")
        relative_times = (
            np.arange(n_timepoints, dtype=float) - stimulus_onset_index
        ) * sampling_interval

    post_mask = relative_times > 0
    if not np.any(post_mask):
        raise ValueError("There must be at least one timepoint after stimulus onset")
    if np.any(hits[:, relative_times < 0]):
        warnings.warn(
            "At least one pre-stimulus True value was found. These samples are "
            "excluded, but they conflict with a zero pre-stimulus failure rate.",
            stacklevel=2,
        )

    if not isinstance(consecutive_samples, (int, np.integer)):
        raise TypeError("consecutive_samples must be an integer")
    if consecutive_samples < 1:
        raise ValueError("consecutive_samples must be at least 1")

    post_hits = hits[:, post_mask]#only consider positive value in time
    post_times = relative_times[post_mask]#only consider positive value in time
    if consecutive_samples > post_hits.shape[1]:
        raise ValueError(
            "consecutive_samples exceeds the number of post-stimulus samples"
        )
    if consecutive_samples == 1:
        qualifying = post_hits
    else:
        windows = np.lib.stride_tricks.sliding_window_view(
            post_hits, window_shape=consecutive_samples, axis=1
        )
        qualifying = windows.all(axis=-1)

    event = qualifying.any(axis=1)
    first_index = np.argmax(qualifying, axis=1)
    observed_time = np.full(n_trials, post_times[-1], dtype=float)
    observed_time[event] = post_times[first_index[event]]

    if trial_ids is None:
        trial_ids_array = np.arange(n_trials)
    else:
        trial_ids_array = np.asarray(trial_ids)
        if trial_ids_array.ndim != 1 or len(trial_ids_array) != n_trials:
            raise ValueError("trial_ids must contain one value per array row")

    output = pd.DataFrame(
        {
            trial_col: trial_ids_array,
            "time": observed_time,
            "event": event.astype(int),
        }
    )
    if subject_ids is not None:
        subject_ids_array = np.asarray(subject_ids)
        if subject_ids_array.ndim != 1 or len(subject_ids_array) != n_trials:
            raise ValueError("subject_ids must contain one value per array row")
        output[subject_col] = subject_ids_array
    return output,qualifying

# if __name__ == "__main__":
#     source_path = "Z:/DATA/experiment_Locust_PreferenceArena/sleap_hdf_new"
#     overwrite_existing_tracking = True
#     folder_name = "gregarisation"
#     hdf_file_reclocation(source_path, folder_name, overwrite_existing_tracking)

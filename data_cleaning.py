import pandas as pd
import numpy as np


def sorting_trial_info(stim_info, visual_paradigm_name="coherence", exp_place="Zball"):
    stim_info = stim_info.reset_index()
    raw_column_number = stim_info.shape[1]
    col_num = [2, 4, 5, 7, 11, 13, 15, 17]
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
    for i, j in zip(col_num, col_name):
        stim_info[["tmp", j]] = stim_info.iloc[:, i].str.split("=", expand=True)
        stim_info[j] = pd.to_numeric(stim_info[j])
    if exp_place == "Zball":
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
    stim_info.drop(columns=stim_info.columns[0 : raw_column_number + 1], inplace=True)
    if visual_paradigm_name.endswith("densities"):
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

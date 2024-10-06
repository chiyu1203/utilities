##this is a code created by Sercan Sayin.

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy import signal
import operator
import scipy
from scipy.signal import find_peaks
import math
from math import atan2, degrees
from scipy.ndimage import gaussian_filter

# import seaborn as sns
from scipy.spatial import distance
import pandas as pd

TestAngle = 50  # angle between the posts
TestDist = 3.0  # distance to the locust
# bodylength = 0.12


def diskretize(
    x, y, bodylength
):  # discretize data into equidistant points, using body lengts (https://stackoverflow.com/questions/19117660/how-to-generate-equispaced-interpolating-values)
    tol = bodylength  # 10cm ,roughly 2BL
    i, idx = 0, [0]
    while i < len(x) - 1:
        total_dist = 0
        for j in range(i + 1, len(x)):
            total_dist = math.sqrt((x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2)
            if total_dist > tol:
                idx.append(j)
                break
        i = j + 1
        # print(i)

    return idx


""" 
def diskretizePanda(df): 
    from scipy.spatial.distance import pdist
    # define the distance threshold
    distance_threshold = 0.12
    
    # filter the dataframe to only include the x and y values
    df_filtered = df[['fx', 'fy']]
    
    # compute the distance matrix row-by-row
    dist_array = pdist(df_filtered, metric='euclidean')
    
    # compute the minimum distance between each pair of points
    min_distance = np.min(dist_array[np.nonzero(dist_array)])
    
    # calculate the number of equally distant points
    num_points = int(np.ceil(np.max(dist_array) / distance_threshold))
    
    # create an array of distances to select the points
    dist_array = np.linspace(0, min_distance, num_points)
    
    # iterate over the distances, selecting the point closest to each distance
    selected_rows = []
    for dist in dist_array:
        row_idx = np.argmin(np.abs(dist_array - dist))
        selected_rows.append(row_idx)
    
    # filter the original dataframe using the selected rows
    df_selected = df.iloc[selected_rows]
    
    return df_selected
"""


def removeNoiseVR(X, Y):

    Xraw = X
    Yraw = Y

    a = np.array(calc_eucledian(X, Y))
    indikes = np.argwhere(a > 0.4)

    NewX = np.delete(np.diff(X), indikes.T)
    NewY = np.delete(np.diff(Y), indikes.T)
    X = np.nancumsum(NewX)
    Y = np.nancumsum(NewY)

    # a = ListAngles(X, Y)
    # a = np.abs(np.gradient(a))
    # indikes = np.argwhere(a < 0.00000001)

    # NewX = np.delete(np.diff(X), indikes.T)
    # NewY = np.delete(np.diff(Y), indikes.T)
    # X = np.cumsum(NewX)
    # Y = np.cumsum(NewY)

    trackingloss = len(X) / len(Xraw)

    return trackingloss, X, Y


def removePandaNoiseVR(df):

    X = df["fx"].values
    Y = df["fy"].values

    Xraw = X
    Yraw = Y

    # plt.plot(X,Y)
    # plt.show()

    a = np.array(calc_eucledian(X, Y))
    indikes = np.argwhere(a > 0.01)

    plt.plot(indikes.T[0])
    plt.show()

    if len(list(indikes.T[0])) == 0:
        pass
    else:
        NewX = np.delete(np.diff(X), indikes.T)
        NewY = np.delete(np.diff(Y), indikes.T)
        X = np.cumsum(NewX)
        Y = np.cumsum(NewY)

        # dfzero = df.iloc[0,:]

        df = df.diff()
        df = df[1:].reset_index(drop=True)
        # print(df)
        df = df.drop(index=list(indikes.T[0]), axis=0)
        df = df.reset_index(drop=True)
        df = df.cumsum()
        # df = df + dfzero

    a = ListAngles(X, Y)
    a = np.abs(np.gradient(a))
    indikes = np.argwhere(a < 0.0001)

    plt.plot(indikes.T[0])
    plt.show()

    if len(list(indikes.T[0])) == 0:
        pass
    else:
        NewX = np.delete(np.diff(X), indikes.T)
        NewY = np.delete(np.diff(Y), indikes.T)
        X = np.cumsum(NewX)
        Y = np.cumsum(NewY)
        # dfzero = df.iloc[0,:]

        df = df.diff()
        # print(df)
        df = df[1:].reset_index(drop=True)
        # print(df)
        df = df.drop(index=list(indikes.T[0]), axis=0)
        print(df)
        plt.plot(df.index.values)
        plt.show()
        df = df.reset_index(drop=True)
        df = df.cumsum()
        # df = df + dfzero

    trackingloss = len(X) / len(Xraw)

    print("cleaning done")

    return trackingloss, df


def removePandaNoiseVR2(df):

    X = df["fx"].values
    Y = df["fy"].values

    Xraw = X
    Yraw = Y

    # plt.plot(X,Y)
    # plt.show()

    a = np.array(calc_eucledian(X, Y))
    indikes = np.argwhere(a > 0.1)

    plt.plot(indikes.T[0])
    plt.show()

    if len(list(indikes.T[0])) == 0:
        pass
    else:
        NewX = np.delete(np.diff(X), indikes.T)
        NewY = np.delete(np.diff(Y), indikes.T)
        X = np.cumsum(NewX)
        Y = np.cumsum(NewY)

        # dfzero = df.iloc[0,:]

        df = df.diff()
        df = df[1:].reset_index(drop=True)
        # print(df)
        df = df.drop(index=list(indikes.T[0]), axis=0)
        df = df.reset_index(drop=True)
        df = df.cumsum()
        # df = df + dfzero

    a = ListAngles(X, Y)
    a = np.abs(np.gradient(a))
    indikes = np.argwhere(a < 0.01)

    plt.plot(indikes.T[0])
    plt.show()

    if len(list(indikes.T[0])) == 0:
        pass
    else:
        NewX = np.delete(np.diff(X), indikes.T)
        NewY = np.delete(np.diff(Y), indikes.T)
        X = np.cumsum(NewX)
        Y = np.cumsum(NewY)
        # dfzero = df.iloc[0,:]

        df = df.diff()
        # print(df)
        df = df[1:].reset_index(drop=True)
        # print(df)
        df = df.drop(index=list(indikes.T[0]), axis=0)
        # print(df)
        # plt.plot(df.index.values)
        # plt.show()
        df = df.reset_index(drop=True)
        df = df.cumsum()
        # df = df + dfzero

    trackingloss = len(X) / len(Xraw)

    print("cleaning done")

    return trackingloss, df


# Calculate Euclidean distance
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# Calculate angular change
def angular_change(x1, y1, x2, y2):
    angle1 = np.arctan2(y1, x1)
    angle2 = np.arctan2(y2, x2)
    return np.abs(angle1 - angle2)


def removePandaNoiseVR3(df):
    # Filter DataFrame according to Euclidean distance
    df["euclidean_distance"] = euclidean_distance(
        df["fx"].shift(), df["fy"].shift(), df["fx"], df["fy"]
    )
    mask1 = df["euclidean_distance"] < 0.1
    df_filtered_1 = df[mask1].drop(columns=["euclidean_distance"])

    # Calculate angular change and apply the gradient filter
    df_filtered_1["angular_change"] = angular_change(
        df_filtered_1["fx"].shift(),
        df_filtered_1["fy"].shift(),
        df_filtered_1["fx"],
        df_filtered_1["fy"],
    )
    mask2 = df_filtered_1["angular_change"] > 0.01
    df_filtered_2 = df_filtered_1[mask2].drop(columns=["angular_change"])

    plt.plot(df_filtered_2.index.values)
    plt.show()

    # Find continuous chunks in the filtered DataFrame
    df_filtered_2["group"] = (df_filtered_2.index.to_series().diff() > 1).cumsum()

    # Filter continuous chunks with more than 60 rows and store them in a list
    grouped = df_filtered_2.groupby("group")
    filtered_chunks = []
    for _, group in grouped:
        if len(group) > 60:
            group = group.drop(columns=["group"]).reset_index(drop=True)
            filtered_chunks.append(group)

    # filtered_chunks is a list of DataFrames, each representing a continuous chunk with more than 60 rows

    return filtered_chunks
    # filtered_chunks is a list of DataFrames, each representing a continuous chunk with more than 60 rows


def split_continuous_chunks(arr):
    result = []
    current_chunk = [arr[0]]

    for i in range(1, len(arr)):
        if arr[i] == arr[i - 1] + 1:
            current_chunk.append(arr[i])
        else:
            if len(current_chunk) > 1:
                result.append(current_chunk)
            current_chunk = [arr[i]]

    if len(current_chunk) > 1:
        result.append(current_chunk)
    results = [np.array(r) for r in result]

    return results


def removePandaNoiseVRbroad(X, Y, df):

    Xraw = X
    Yraw = Y

    # plt.plot(X,Y)
    # plt.show()

    a = np.array(calc_eucledian(X, Y))
    indikes = np.argwhere(a > 0.01)

    ind = indikes.T[0]
    print(ind)

    if len(list(indikes.T[0])) == 0:
        pass
    else:
        # NewX = np.delete(np.diff(X), indikes.T)
        # NewY = np.delete(np.diff(Y), indikes.T)
        # X = np.cumsum(NewX)
        # Y = np.cumsum(NewY)
        # dfzero = df.iloc[0,:]

        indikes = split_continuous_chunks(ind)
        indikes = [item for sublist in indikes for item in sublist]

        df = df.diff()
        df = df[1:].reset_index(drop=True)
        # print(df)
        df = df.drop(index=list(indikes), axis=0)
        df = df.reset_index(drop=True)
        df = df.cumsum()
        # df = df + dfzero

    a = ListAngles(X, Y)
    a = np.abs(np.gradient(a))
    indikes = np.argwhere(a < 0.0001)
    ind = indikes.T[0]
    print(ind)

    if len(list(indikes.T[0])) == 0:
        pass
    else:

        # NewX = np.delete(np.diff(X), indikes.T)
        # NewY = np.delete(np.diff(Y), indikes.T)
        # X = np.cumsum(NewX)
        # Y = np.cumsum(NewY)
        # dfzero = df.iloc[0,:]

        indikes = split_continuous_chunks(ind)
        indikes = [item for sublist in indikes for item in sublist]

        df = df.diff()
        df = df[1:].reset_index(drop=True)
        # print(df)
        df = df.drop(index=list(indikes), axis=0)
        df = df.reset_index(drop=True)
        df = df.cumsum()
        # df = df + dfzero

    trackingloss = len(X) / len(Xraw)

    print("cleaning done")

    return trackingloss, df


def AngleBtw2Points(pointA, pointB):
    changeInX = pointB[0] - pointA[0]
    changeInY = pointB[1] - pointA[1]
    return atan2(changeInY, changeInX)


def AngleOverList(X, Y):  # possibly wrong!!!
    ang = []
    for i in range(len(X) - 1):
        changeInX = X[i + 1] - X[i]
        changeInY = Y[i + 1] - Y[i]

        a = abs(atan2(changeInY, changeInX))
        ang.append(a)

    return np.median(ang)


def ListAngles(X, Y):
    ang = []
    for i in range(len(X) - 1):
        changeInX = X[i + 1] - X[i]
        changeInY = Y[i + 1] - Y[i]

        a = atan2(changeInY, changeInX)
        if a < 0:
            a = a + 2 * np.pi
        ang.append(a)

    return ang


def ListAngles2(X, Y):
    ang = []
    for i in range(len(X) - 1):
        changeInX = X[i + 1] - X[i]
        changeInY = Y[i + 1] - Y[i]

        a = atan2(changeInY, changeInX)
        # if a < 0:
        #    a = a + 2*np.pi
        ang.append(a)

    return ang


def circfilt(
    a, constant
):  # https://stats.stackexchange.com/questions/114842/average-and-standard-deviation-of-timestamps-time-wraps-around-at-midnight/115123#115123

    angs = np.array(a)
    n = len(angs)
    S = np.sum(np.sin(angs))
    C = np.sum(np.cos(angs))

    mu_hat = np.array(math.atan2(S, C))

    R_bar = np.sqrt(S * S + C * C) / n

    # delta_hat = (1 - np.sum(np.cos(2 * (angs-mu_hat)))/n) / (2 * np.sqrt(R_bar))
    delta_hat = (1 - np.sum(np.cos(2 * (angs - mu_hat))) / n) / (2 * R_bar * R_bar)

    low = mu_hat - constant * delta_hat
    high = mu_hat + constant * delta_hat

    fl = angs > low
    fh = angs < high

    filtered = angs[fl]
    filtered = angs[fh]
    return filtered


def meanAngleOverList(X, Y):
    ang = []
    for i in range(len(X) - 1):
        changeInX = X[i + 1] - X[i]
        changeInY = Y[i + 1] - Y[i]

        a = atan2(changeInY, changeInX)
        ang.append(a)

    return np.mean(ang)


def rotate_vector(x, y, angle):

    co = np.cos(angle)
    si = np.sin(angle)

    rotatedx = []
    rotatedy = []

    for i in range(len(x)):
        rotatedx.append(x[i] * co - y[i] * si)
        rotatedy.append(x[i] * si + y[i] * co)

    return rotatedx, rotatedy


def rotate_vector2(x, y, angle):
    co = np.cos(angle)
    si = np.sin(angle)

    rotatedx = x * co - y * si
    rotatedy = x * si + y * co

    return rotatedx, rotatedy


def rotatePanda(df, angle):

    rotated_x_values = []
    rotated_y_values = []

    # Assuming the first 64 columns are x values and the next 64 columns are y values

    x_values = df.iloc[:, :64].to_numpy()
    y_values = df.iloc[:, 64:128].to_numpy()

    rotated_x_values, rotated_y_values = rotate_vector(x_values, y_values, angle)

    # Combine the rotated x and y values into a single DataFrame
    rotated_df = pd.DataFrame(
        np.hstack([rotated_x_values, rotated_y_values]), columns=df.columns
    )

    return rotated_df


def rotatePanda_heading(df):
    # Calculate the cumulative theta position value for each row (assuming you already have a 'cumulative_theta' column)
    # Replace this part with your actual code to obtain the 'cumulative_theta' column
    df["theta"] = np.arctan2(df["fx"], df["fy"])
    df["delta_theta"] = df["theta"].diff()
    df["cumulative_theta"] = df["delta_theta"].cumsum()

    # Prepare a dictionary to store the rotated x and y columns
    rotated_columns = {}

    # Rotate each x, y pair using the cumulative theta position value
    for i in range(1, 65):
        x_col = f"x{i}"
        y_col = f"y{i}"
        x_rot_col = f"x_rot{i}"
        y_rot_col = f"y_rot{i}"

        rotated_columns[x_rot_col] = df[x_col] * np.cos(df["cumulative_theta"]) - df[
            y_col
        ] * np.sin(df["cumulative_theta"])
        rotated_columns[y_rot_col] = df[x_col] * np.sin(df["cumulative_theta"]) + df[
            y_col
        ] * np.cos(df["cumulative_theta"])

    # Convert the dictionary to a dataframe and join it to the original dataframe
    df_rotated = pd.DataFrame(rotated_columns)
    df = pd.concat([df, df_rotated], axis=1)

    return df


def rotate_vector_insquare(X, Y, angle):

    rotatedx, rotatedy = [], []

    lngth = 30000
    l = [lngth * 0, lngth * 1, lngth * 2, lngth * 3]

    for i in l:

        x, y = rotate_vector(
            X[i : i + lngth], Y[i : i + lngth], angle + 1.57 * i / 30000
        )
        rotatedx.append(x)
        rotatedy.append(y)

    rx = [item for sublist in rotatedx for item in sublist]
    ry = [item for sublist in rotatedy for item in sublist]

    rotatedx, rotatedy = np.cumsum(rx) * -1, np.cumsum(ry) * -1

    return rotatedx, rotatedy


def dataHandler(data):
    # print(len(data[0]))
    if len(data) <= 37500:
        resampled = scipy.signal.resample(data, 37500)
    else:
        resampled = scipy.signal.resample(data, 60000)

    resampled = resampled - resampled[0]

    # print(len(resampled[0]))
    return resampled


def dataHandler_old(array0, array1, target):

    sampledx = scipy.signal.resample(array0, target)
    sampledy = scipy.signal.resample(array1, target)

    diffarx = np.diff(sampledx)
    diffary = np.diff(sampledy)
    """
    plt.plot(diffarx)
    plt.title("X")
    plt.show()
    
    plt.plot(diffary)
    plt.title("Y")
    plt.show()
    """
    diffarXY = np.stack((diffarx, diffary), axis=-1)

    indikes = np.argwhere((diffarXY < -0.1) | (diffarXY > 0.1))

    NewX = np.delete(diffarx, indikes.T)
    NewY = np.delete(diffary, indikes.T)

    """
    plt.plot(diffarx[-10000:])
    plt.title("oldX")
    plt.show()
    
    plt.plot(NewX[-10000:])
    plt.title("NewX")
    plt.show()

    plt.plot(diffary[-10000:])
    plt.title("oldY")
    plt.show()
    
    plt.plot(NewY[-10000:])
    plt.title("NewY")
    plt.show()    
    """
    #    return sampledx,sampledy
    return np.cumsum(NewX), np.cumsum(NewY)


# https://stackoverflow.com/questions/29156532/python-baseline-correction-library
def correctBaseline(y, lam, p, niter=10):
    L = len(y)
    D = scipy.sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = scipy.sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = scipy.sparse.spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def dataHandler_old_noresample(array0, array1):

    diffarx = np.diff(array0)
    diffary = np.diff(array1)
    """
    plt.plot(diffarx)
    plt.title("X")
    plt.show()
    
    plt.plot(diffary)
    plt.title("Y")
    plt.show()
    """
    diffarXY = np.stack((diffarx, diffary), axis=-1)

    indikes = np.argwhere((diffarXY < -0.1) | (diffarXY > 0.1))

    NewX = np.delete(diffarx, indikes.T)
    NewY = np.delete(diffary, indikes.T)

    """
    plt.plot(diffarx[-10000:])
    plt.title("oldX")
    plt.show()
    
    plt.plot(NewX[-10000:])
    plt.title("NewX")
    plt.show()

    plt.plot(diffary[-10000:])
    plt.title("oldY")
    plt.show()
    
    plt.plot(NewY[-10000:])
    plt.title("NewY")
    plt.show()    
    """
    #    return sampledx,sampledy
    return np.cumsum(NewX), np.cumsum(NewY)


def removeNoise(x, y):

    NewX = scipy.ndimage.gaussian_filter1d(x, sigma=2)
    NewY = scipy.ndimage.gaussian_filter1d(y, sigma=2)

    return NewX, NewY


def calc_angle(List1, List2):
    w = 0
    theta = []
    dot = []
    while w < len(List1) - 1:

        Vector1 = List1[w + 1] - List1[w]
        Vector2 = List2[w + 1] - List2[w]

        UnitVector1 = Vector1 / np.linalg.norm(Vector1)
        UnitVector2 = Vector2 / np.linalg.norm(Vector2)

        dotproduct = np.dot(UnitVector1, UnitVector2)
        angle = np.arccos(dotproduct)

        dot.append(dotproduct)
        theta.append(angle)

        w = w + 1

    # plt.plot(theta)
    # plt.show

    return np.mean(theta)


def calc_eucledian(xx1, yy1):  # calculates spontaneous euclidian distance over time
    w = 0
    dist = []
    while w < len(xx1) - 1:

        a = np.array((xx1[w + 1], yy1[w + 1]))
        b = np.array((xx1[w], yy1[w]))

        euc = np.linalg.norm(a - b)
        # euc = distance.euclidean(b,a)
        dist.append(euc)
        w = w + 1

    return dist


"""
import pandas as pd
import numpy as np

# Calculate Euclidean distance
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Calculate angular change
def angular_change(x1, y1, x2, y2):
    angle1 = np.arctan2(y1, x1)
    angle2 = np.arctan2(y2, x2)
    return np.abs(angle1 - angle2)

def cleanpanda(df):
    # Filter DataFrame according to Euclidean distance
    df['euclidean_distance'] = euclidean_distance(df['fx'].shift(), df['fy'].shift(), df['fx'], df['fy'])
    #print(df['euclidean_distance'] )
    #plt.plot(df['euclidean_distance'].values)
    mask1 = (df['euclidean_distance'] < 0.01)
    df[:,:-1] = df[:,:-1].diff()
    
    df_filtered_1 = df[mask1].drop(columns=['euclidean_distance'])
    df_filtered_1 = df_filtered_1.reset_index(drop=True)
    df_filtered_1 = df_filtered_1.cumsum()
    
    # Filter DataFrame according to angular change
    df_filtered_1['angular_change'] = angular_change(df_filtered_1['fx'].shift(), df_filtered_1['fy'].shift(), df_filtered_1['fx'], df_filtered_1['fy'])
    
    #print(df_filtered_1['angular_change'])
    plt.plot(df_filtered_1['angular_change'].values)
    plt.ylim(0,0.001)
    plt.show()
    mask2 = (df_filtered_1['angular_change'] < 0.0001)
    #df_filtered_1 = df_filtered_1.diff()
    df_filtered_2 = df_filtered_1[mask2].drop(columns=['angular_change'])
    #df_filtered_2 = df_filtered_2.cumsum()
    # Return the new DataFrame
    filtered_df = df_filtered_2.reset_index(drop=True)
    
    return filtered_df
"""

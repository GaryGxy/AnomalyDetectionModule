import numpy as np


def sliding_window(df, percentage, threshold):
    """
    This function implements a sliding window algorithm.
    The function takes three parameters as inputs - a pandas dataframe, a numeric value and
    a numeric threshold.
    It creates a sliding window with the size according to the percentage parameter and go over
    the input dataset.
    The function returns two tuples stand for the upper and lower band.
    """
    window_percentage = percentage
    k = int(len(df["value"])*(window_percentage/2/100))
    n = len(df["value"])
    get_bands = lambda data: (np.mean(data)+np.nanquantile(data, threshold),
                              np.mean(data)-np.nanquantile(data, threshold))
    bands = [get_bands(df["value"].loc[range(0 if i-k < 0 else i-k, i+k if i+k < n else n)]) for i in range(0, n)]
    upper, lower = zip(*bands)
    return upper, lower


def anomaly_detect(df, percentage, threshold):
    """
    This function detects the abnormal data points of the given dataset.
    The function takes three parameters as inputs - a pandas dataframe, a numeric value and
    a numeric threshold.
    The function returns a list containing all the outlier that are either higher than the
    upper band or lower than the lower band.
    """
    outlier = []
    upper, lower = sliding_window(df, percentage, threshold)
    for i in range(0, len(upper)):
        if df["value"][i] > upper[i] or df["value"][i] < lower[i]:
            outlier.append(df.iloc[i])
    return outlier


def summary(outlier):
    """
    This function prints out the result of anomaly detection.
    The function takes a list of predicted outliers as input
    and print out the summary of the result.
    """
    count = 0
    for x in outlier:
        if x["Label"] == 1:
            count += 1
            print("Anomaly detected : " + str(x) + "\n")
    print("There are " + str(count) + " true positive in " + str(len(outlier)) + " predicted outliers" + "\n")

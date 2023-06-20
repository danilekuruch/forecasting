import sys
import logging
from tkinter import filedialog
import numpy as np
import seaborn as sns
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from scipy import stats
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def log_separator():
    """Prints a separator in the log."""
    log.info("\n")


def get_cmd_args():
    """
    Collects command arguments in one dictionary object

    Returns:
        dict: a dictionary object, which includes command arguments
    """
    if len(sys.argv) > 1:
        argv = sys.argv[1:]
        args_dict = {}
        for i in range(0, len(argv), 2):
            args_dict[argv[i]] = argv[i + 1]
        return args_dict
    return {}


def get_or_default(args, field, default):
    """
    Takes from "args" a necessary field or returns a default value

    Args:
        args (dict): a dictionary of command args
        field (string): a command argument
        default (any): a value, which returns, if "args" parameter does not have a necessary field

    Returns:
        any: value from "args" object of "field" key
    """
    if field in args:
        return args[field]
    return default


def read_csv_file(file_path, columns):
    """
    Reads a CSV file and returns a DataFrame.

    Args:
        columns: List of column names to read from the CSV file.

    Returns:
        The DataFrame containing the data read from the CSV file.
    """
    try:
        if not file_path:
            file_path = filedialog.askopenfilename()
        return pd.read_csv(file_path, usecols=columns)
    except FileNotFoundError:
        log.error("File is not found.")
        sys.exit()


def mape(y_actual, y_predicted):
    """
    Calculates the Mean Absolute Percentage Error (MAPE).

    Args:
        y_actual: Array-like object of actual values.
        y_predicted: Array-like object of predicted values.

    Returns:
        The MAPE value.
    """
    numerator = np.where(
        y_actual == 0, y_predicted - y_actual, (y_predicted - y_actual) / y_actual
    )
    return np.median(np.abs(numerator)) * 100


def u_1(y_actual, y_predicted):
    """
    Calculates the U1 metric.

    Args:
        y_actual: Array-like object of actual values.
        y_predicted: Array-like object of predicted values.

    Returns:
        The U1 value.
    """
    numerator = y_predicted - y_actual
    numerator = np.sqrt(np.median(np.square(numerator)))

    denominator = np.sqrt(np.median(np.square(y_predicted))) + np.sqrt(
        np.median(np.square(y_actual))
    )

    return numerator / denominator


def u_2(y_actual, y_predicted):
    """
    Calculates the U2 metric.

    Args:
        y_actual: Array-like object of actual values.
        y_predicted: Array-like object of predicted values.

    Returns:
        The U2 value.
    """
    numerator = np.where(
        y_actual[:-1] == 0,
        y_predicted[1:] - y_actual[1:],
        (y_predicted[1:] - y_actual[1:]) / y_actual[:-1],
    )
    denominator = np.where(
        y_actual[:-1] == 0,
        y_actual[1:] - y_actual[:-1],
        (y_actual[1:] - y_actual[:-1]) / y_actual[:-1],
    )

    return np.sqrt(np.sum(np.square(numerator)) / np.sum(np.square(denominator)))


def build_time_series(data_frame, length, index, name):
    """
    Builds and plots a time series.

    Args:
        data_frame: DataFrame containing the data.
        time_series_length: The length of the processing time series data.
        time_series_index: Index of the specific time series column in the DataFrame.
        name: Name of the time series.
    """
    plt.figure(label="Time Series Plot")
    plt.plot(
        data_frame.values[:length, index],
        label=name,
    )

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Time Series")
    plt.legend()

    log.info("Data frame values:\n%s", data_frame)

    plt.tight_layout()
    plt.show()


def perform_stationarity(time_serie, window_size):
    """
    Computes and plots the rolling mean and standard deviation to assess stationarity.

    Args:
        time_series: Array-like object representing the time series data.
        window_size: Size of the rolling window for computing the mean and standard deviation.
    """
    rolling_mean = np.convolve(
        time_serie, np.ones(window_size) / window_size, mode="valid"
    )
    rolling_std = np.std(
        np.lib.stride_tricks.sliding_window_view(time_serie, window_size), axis=1
    )

    plt.figure(label="Assess Stationarity")
    plt.plot(time_serie, color="blue", label="Original")
    plt.plot(
        np.pad(rolling_mean, (window_size - 1, 0), mode="constant"),
        color="red",
        label="Mean",
    )
    plt.plot(
        np.pad(rolling_std, (window_size - 1, 0), mode="constant"),
        color="black",
        label="Std",
    )
    plt.legend(loc="best")
    plt.title("Mean & Standard Deviation")
    plt.show(block=False)


def perform_stl_decomposition(time_serie, length, index):
    """
    Performs STL (Seasonal and Trend decomposition using Loess)
        decomposition on a given time series.

    Args:
        time_series: Array-like object representing the time series data.
        time_series_length: The length of the processing time series data.
        time_series_index: Index of the specific time series to decompose within the given array.
    """
    log_time_series = np.log(time_serie[:length, index] + 1)
    res_robust = STL(log_time_series, period=3, robust=True).fit()
    res_non_robust = STL(log_time_series, period=3, robust=False).fit()

    axs = res_robust.plot().get_axes()
    components = ["trend", "seasonal", "resid"]
    for axes, component in zip(axs[1:], components):
        series = getattr(res_non_robust, component)
        if component == "resid":
            axes.plot(series, marker="o", linestyle="none")
        else:
            axes.plot(series)
            if component == "trend":
                axes.legend(["Robust", "Non-robust"], frameon=False)

    for name, res in [("Non-robust", res_non_robust), ("Robust", res_robust)]:
        trend_mape = mape(log_time_series, res.trend)
        trend_u1 = u_1(log_time_series, res.trend)
        trend_u2 = u_2(log_time_series, res.trend)
        trend_seasonality_mape = mape(log_time_series, res.trend + res.seasonal)
        trend_seasonality_u1 = u_1(log_time_series, res.trend + res.seasonal)
        trend_seasonality_u2 = u_2(log_time_series, res.trend + res.seasonal)

        log_separator()
        log.info("%s trend MAPE: %s", name, trend_mape)
        log.info("%s trend U1: %s", name, trend_u1)
        log.info("%s trend U2: %s", name, trend_u2)
        log.info("%s trend and seasonality MAPE: %s", name, trend_seasonality_mape)
        log.info("%s trend and seasonality U1: %s", name, trend_seasonality_u1)
        log.info("%s trend and seasonality U2: %s", name, trend_seasonality_u2)

    plt.show()


def perform_stl_forecasting(time_serie, length, index, steps, is_robust):
    """
    Performs STL forecasting on a time series and plots the results.

    Args:
        time_series: Array-like object representing the time series data.
        time_series_length: The length of the processing time series data.
        time_series_index: Index of the time series column.
        steps: Number of future time steps to forecast.
        is_robust: Indicates whether to use robust STL decomposition or not.
    """
    forecast_type = "Robust" if is_robust else "Non-Robust"

    forecast = (
        STLForecast(
            np.log(time_serie[:length, index] + 1),
            ARIMA,
            model_kwargs={"order": (2, 1, 0)},
            period=5,
            robust=False,
        )
        .fit()
        .forecast(steps)
    )

    plt.figure(figsize=(10, 6), label=f"STL Forecasting: {forecast_type}")
    plt.plot(
        np.log(time_serie[: length + steps, index] + 1),
        label="Original",
        color="blue",
    )
    plt.plot(
        range(length, length + steps),
        forecast,
        label="Forecast",
        color="red",
    )

    plt.xlabel("Time")
    plt.ylabel("Log Scale")
    plt.title(f"STL Forecasting: {forecast_type}")
    plt.legend()
    plt.show()


def check_normality_jarque_bera(data_frame_values, length, index):
    """
    Performs the Jarque-Bera to check the normality of data.

    Args:
        data (list or numpy array): The data to be tested.

    Returns:
        bool: True if the data follows a normal distribution, False otherwise.
    """
    jarque_bera_test = stats.jarque_bera(data_frame_values[:length, index])

    is_normal = jarque_bera_test[1] > 0.05

    log.info(
        "Testing normality: performs the Jarque-Bera test:\t%s",
        jarque_bera_test,
    )

    log.info("Testing normality - result:\t%s", "Normal" if is_normal else "Not normal")

    return is_normal


def check_stationarity_dickey_fuller(data_frame_values, length, index):
    """
    Performs the Augmented Dickey-Fuller test to check the stationarity of a time series.

    Args:
        data_frame (pandas DataFrame): The DataFrame object containing the time series data.
        time_serie_index (int): The index of the time series column in the DataFrame.

    Returns:
        bool: True if the time series is stationary, False otherwise.
    """
    dickey_fuller_test = adfuller(data_frame_values[:length, index])

    is_stationary = dickey_fuller_test[1] < 0.05

    log.info(
        "Testing stationarity: perform the Augmented Dickey-Fuller test:\t%s",
        dickey_fuller_test,
    )
    log.info(
        "Testing stationarity - result:\t%s",
        "Stationary" if is_stationary else "Not stationary",
    )

    return is_stationary


def main():
    """
    Main function for executing the forecasting process.
    """
    args = get_cmd_args()

    time_serie_name = get_or_default(args, "--ts-name", "NA_Sales")
    time_serie_index = int(get_or_default(args, "--ts-index", 0))
    time_serie_length = int(get_or_default(args, "--ts-length", 200))
    forecast_steps = int(get_or_default(args, "--forecast-steps", 6))
    path = get_or_default(args, "--path", None)

    data_frame = read_csv_file(path, columns=[time_serie_name])
    data_frame_values = data_frame.values

    build_time_series(
        data_frame=data_frame,
        length=time_serie_length,
        index=time_serie_index,
        name=time_serie_name,
    )

    register_matplotlib_converters()
    sns.set_style("darkgrid")
    plt.rc("figure", figsize=(16, 12))
    plt.rc("font", size=13)

    log_separator()

    check_normality_jarque_bera(data_frame_values, time_serie_length, time_serie_index)

    check_stationarity_dickey_fuller(
        data_frame_values, time_serie_length, time_serie_index
    )

    perform_stationarity(
        time_serie=data_frame_values[:time_serie_length, time_serie_index],
        window_size=12,
    )

    perform_stl_decomposition(
        time_serie=data_frame_values,
        length=time_serie_length,
        index=time_serie_index,
    )

    perform_stl_forecasting(
        time_serie=data_frame_values,
        length=time_serie_length,
        index=time_serie_index,
        steps=forecast_steps,
        is_robust=True,
    )

    perform_stl_forecasting(
        time_serie=data_frame_values,
        length=time_serie_length,
        index=time_serie_index,
        steps=forecast_steps,
        is_robust=False,
    )


if __name__ == "__main__":
    main()

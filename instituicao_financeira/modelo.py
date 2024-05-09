from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def load_data_from_excel(path_file: str, tab: str) -> DataFrame | dict[Any, DataFrame]:
    """
    Load data from an Excel file.

    Args:
        path_file (str): The path to the Excel file.
        tab (str): The name of the tab in the Excel file.

    Returns:
        DataFrame | dict[Any, DataFrame]: The loaded data from the Excel file.
    """
    # Replace backslashes with forward slashes in the path
    path_file = path_file.replace('\\', '/')

    # Read the Excel file and return the loaded data
    data_base = pd.read_excel(path_file, tab)
    return data_base


def generate_scatter_graph(df):
    """
    Generate a scatter graph with data from a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.

    Returns:
        None
    """
    graph = px.scatter(x=df.iloc[:, 0].values, y=df.iloc[:, 1].values, opacity=0.65)
    graph.update_layout(title="Crédito x Salário", xaxis_title='Salário', yaxis_title='Limite Crédito')
    graph.write_html(
        "H:/python/machine_learning/instituicao_financeira/graphics/graphic_scattered_salario_credito.html",
        auto_open=True)


def generate_heatmap(df):
    """
    Generate a heatmap graphic using Plotly Express.

    Parameters:
        df (pandas.DataFrame): The dataframe containing the data for the heatmap.

    Returns:
        None
    """
    heatmap_graphic = px.imshow(df, text_auto=True)
    heatmap_graphic.write_html("H:/python/machine_learning/instituicao_financeira/graphics/graphic_heatmap.html",
                               auto_open=True)


def split_data(X_AXIS, Y_AXIS):
    """
    Split the data into training and testing sets.

    Args:
        X_AXIS: The input features.
        Y_AXIS: The target variable.

    Returns:
        X_TRAIN: The training input features.
        X_TEST: The testing input features.
        Y_TRAIN: The training target variable.
        Y_TEST: The testing target variable.
    """
    from sklearn.model_selection import train_test_split

    # Split the data into training and testing sets
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X_AXIS, Y_AXIS, test_size=0.2)

    return X_TRAIN, X_TEST, Y_TRAIN, Y_TEST


def train_linear_regression_model(X_train, X_test, Y_train, Y_test):
    """
    Trains a linear regression model using the given training data and returns the trained model,
    predictions on the test data, and the mean squared error of the predictions.

    Args:
        X_train (array-like): The training data features.
        X_test (array-like): The test data features.
        Y_train (array-like): The training data labels.
        Y_test (array-like): The test data labels.

    Returns:
        model (LinearRegression): The trained linear regression model.
        predictions (array-like): The predictions on the test data.
        accuracy (float): The mean squared error of the predictions.
    """
    model = LinearRegression()
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    accuracy = mean_squared_error(Y_test, predictions)
    return model, predictions, accuracy


def graphic_model(input_train, input_test, target_train, target_test, predictions):
    """
    Plot a graph to visualize the training and testing data along with the predictions.

    Args:
        input_train (ndarray): Training input data.
        input_test (ndarray): Testing input data.
        target_train (ndarray): Training target data.
        target_test (ndarray): Testing target data.
        predictions (ndarray): Predicted values.

    Returns:
        None
    """
    fig = go.Figure([
        go.Scatter(x=input_train.squeeze(), y=target_train.squeeze(), name="train", mode='markers'),
        go.Scatter(x=input_test.squeeze(), y=target_test.squeeze(), name="test", mode='markers'),
        go.Scatter(x=input_test.squeeze(), y=predictions.squeeze(), mode='lines', name='prediction')
    ])
    fig.show()


DF = load_data_from_excel("H:\python\machine_learning\instituicao_financeira\data_base\BaseDados_RegressaoLinear.xlsx",
                          'Plan1')

# X -> Wage | Y -> Credit Limit
X_AXIS = DF.iloc[:, 0].values.reshape(-1, 1)
Y_AXIS = DF.iloc[:, 1].values.reshape(-1, 1)

COR_X_Y = DF.corr()

X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = split_data(X_AXIS, Y_AXIS)
MODEL, PREVISOS, ACCURACY = train_linear_regression_model(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)

graphic_model(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, PREVISOS)

# GRAPHICS_SCATTER(DF)

# GRAPHIC_HEATMAP(COR_X_Y)

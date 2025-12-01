import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
#-----------------------------------------------------------------------------------------------------
def plot_numerical_data(dataframe):
    numerical_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns

    for column in numerical_columns:
        fig, axis = plt.subplots(2, 1, figsize=(8, 4), gridspec_kw={'height_ratios': [6, 1]})

        # Calculate mean, median, and standard deviation
        mean_val = np.mean(dataframe[column])
        median_val = np.median(dataframe[column])
        std_dev = np.std(dataframe[column])

        # Create a multiple subplots with histograms and box plots
        sns.histplot(ax=axis[0], data=dataframe, kde=True, x=column).set(xlabel=None)
        axis[0].axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label='Mean')
        axis[0].axvline(median_val, color='orange', linestyle='dashed', linewidth=1, label='Median')
        axis[0].axvline(mean_val + std_dev, color='green', linestyle='dashed', linewidth=1, label='Standard Deviation')
        axis[0].axvline(mean_val - std_dev, color='green', linestyle='dashed', linewidth=1)

        sns.boxplot(ax=axis[1], data=dataframe, x=column, width=0.6).set(xlabel=None)
        axis[1].axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label='Mean')
        axis[1].axvline(median_val, color='orange', linestyle='dashed', linewidth=1, label='Median')
        axis[1].axvline(mean_val + std_dev, color='green', linestyle='dashed', linewidth=1)
        axis[1].axvline(mean_val - std_dev, color='green', linestyle='dashed', linewidth=1)

        axis[0].legend()

        fig.suptitle(column)

        # Adjust the layout
        plt.tight_layout()

        # Show the plot
        plt.show()
#-----------------------------------------------------------------------------------------------------
def plot_scatter_heatmaps(dataframe, target_variable):
    numeric_variables = dataframe.select_dtypes(include=['float64', 'int64']).columns
    num_cols = 2
    num_rows = len(numeric_variables) - 1

    fig, axis = plt.subplots(num_rows, num_cols, figsize=(13, 5 * num_rows))

    for i, x_variable in enumerate(numeric_variables):
        # Evitar plotear la variable target
        if x_variable == target_variable:
            continue

        # Gráfico de dispersión
        sns.regplot(ax=axis[i, 0], data=dataframe, x=x_variable, y=target_variable)
        axis[i, 0].set_title(f'Regplot: {x_variable} vs {target_variable}')

        # Mapa de calor
        sns.heatmap(dataframe[[x_variable, target_variable]].corr(), annot=True, fmt=".2f", ax=axis[i, 1])
        axis[i, 1].set_title(f'Heatmap: {x_variable} vs {target_variable}')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajustar la posición del título
    plt.show()

#-----------------------------------------------------------------------------------------------------
def general_heatmap_sorted(dataframe, target_variable):
    # Matriz de correlación completa (solo numéricas por si acaso)
    corr = dataframe.corr(numeric_only=True)
    
    # Ordenar columnas por |corr| con el target
    ordered_cols = corr[target_variable].abs().sort_values(ascending=False).index
    
    corr_sorted = corr.loc[ordered_cols, ordered_cols]

    plt.figure(figsize=(10, 7))
    sns.heatmap(corr_sorted, annot=True, fmt=".2f", cbar=True)
    plt.title(f"Matriz de correlación ordenada por {target_variable}")
    plt.tight_layout()
    plt.show()
#-----------------------------------------------------------------------------------------------------
def outlier_analysis(dataframe,target):
    numerical_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns
    numerical_columns = numerical_columns[numerical_columns != target]
    for column in numerical_columns:
        fig, axis = plt.subplots(figsize=(8, 1.2))
        sns.boxplot(ax=axis, data=dataframe, x=column, width=0.3).set(xlabel=None)
        fig.suptitle(column)
        plt.tight_layout()
        plt.show()
    # Return the describe dataframe    
    return dataframe.describe().T

#-----------------------------------------------------------------------------------------------------
def outliers_summary(dataset,outliers):
    print(f'''the rows with outliers are {len(outliers)}''')
    print(f'''the total rows are {len(dataset)}''')
    print(f'''this represents {round(len(outliers)/len(dataset),2)*100} % of the dataset''')
#-----------------------------------------------------------------------------------------------------
def heatmap_target_only(dataframe, target_variable):
    # Matriz de correlación completa
    corr = dataframe.corr(numeric_only=True)
    
    # Solo la correlación vs target
    corr_target = corr[[target_variable]].sort_values(
        by=target_variable,
        ascending=False
    )

    plt.figure(figsize=(4, 6))
    sns.heatmap(
        corr_target,
        annot=True,
        fmt=".2f",
        cbar=False
    )
    plt.title(f"Correlación con {target_variable}")
    plt.tight_layout()
    plt.show()
#-----------------------------------------------------------------------------------------------------
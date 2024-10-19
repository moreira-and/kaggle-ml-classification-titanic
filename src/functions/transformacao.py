import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def analyze_thresholds(df, numeric_col, target_col, n_clusters=4, plot_title='Taxa de Sobrevivência por Grupo', cluster_plot_title='Clusters de Idade e Sobrevivência'):
    """
    Analyzes the impact of a numeric variable on a target variable using K-means clustering to create thresholds.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        numeric_col (str): The column name for the numeric variable.
        target_col (str): The column name for the target variable.
        n_clusters (int): The number of clusters for K-means.
        plot_title (str): Title for the survival rate plot.
        cluster_plot_title (str): Title for the cluster visualization plot.
        
    Returns:
        thresholds (np.ndarray): The thresholds (centers of clusters).
        survival_rates (pd.DataFrame): DataFrame containing survival rates for each group.
    """
    # Applying K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    df['Group'] = kmeans.fit_predict(df[[numeric_col]])

    # Obtaining the cluster centers
    thresholds = kmeans.cluster_centers_.flatten()
    thresholds.sort()  # Sort the thresholds for better visualization

    # Calculating survival rates by group
    survival_rates = df.groupby('Group')[target_col].mean().reset_index()

    # Displaying the results
    print(f"Média de {target_col} por Grupo:")
    print(survival_rates)

    print("\nThresholds (centros dos clusters):")
    for i, threshold in enumerate(thresholds):
        print(f"Threshold para Grupo {i}: {threshold:.2f}")

    # Visualizing the survival rates by group
    plt.bar(survival_rates['Group'], survival_rates[target_col], color='skyblue')
    plt.title(plot_title)
    plt.xlabel('Grupo')
    plt.ylabel(f'Taxa de {target_col}')
    plt.ylim(0, 1)
    plt.xticks(survival_rates['Group'])
    plt.show()

    # Visualizing the clusters
    plt.scatter(df[numeric_col], df[target_col], c=df['Group'], cmap='viridis', alpha=0.5)
    plt.title(cluster_plot_title)
    plt.xlabel(numeric_col)
    plt.ylabel(target_col)
    plt.show()
    
    return thresholds, survival_rates




# Função para categorizar os dados com base nos thresholds
def categorize_col(df, col, thresholds):
    """
    Categoriza os dados em intervalos com base nos thresholds.
    
    Parameters:
        df (pd.DataFrame): O DataFrame contendo os dados.
        col (str): O nome da coluna que contém as informações a serem clusterizadas.
        thresholds (np.ndarray): Os thresholds usados para a categorização.

    Returns:
        pd.Series: Uma série contendo as categorias correspondentes a cada idade.
    """
    # Criar as categorias
    bins = [-np.inf] + thresholds.tolist() + [np.inf]  # Adiciona -inf e inf para incluir todos os valores
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]  # Cria rótulos para os bins

    # Categorizar os dados
    return pd.cut(df[col], bins=bins, labels=labels, right=False)
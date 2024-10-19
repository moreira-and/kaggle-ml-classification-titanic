import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to count outliers using the IQR method
def count_outliers(data, column):
    """
    
    Retorna um dataframe com os outliers

    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers.shape[0]

def plot_categorical_relationship(df, cat_var1, cat_var2):
    """
    Plota barplots para duas variáveis categóricas, com a segunda variável categórica representada em colunas.
    
    Parameters:
        df (pd.DataFrame): O dataframe contendo as variáveis.
        cat_var1 (str): Nome da primeira variável categórica.
        cat_var2 (str): Nome da segunda variável categórica.
    """
    
    # Verifica se as variáveis estão no DataFrame
    if cat_var1 not in df.columns or cat_var2 not in df.columns:
        raise ValueError(f"As variáveis {cat_var1} ou {cat_var2} não estão no DataFrame.")

    # Cria o gráfico usando catplot
    plt.figure(figsize=(12, 6))
    sns.catplot(
        data=df, 
        x=cat_var1, 
        kind='count', 
        col=cat_var2,  
        height=4, 
        aspect=0.7
    )
    
    plt.subplots_adjust(top=0.8)
    plt.suptitle(f'Relação entre {cat_var1} e {cat_var2}', fontsize=16)
    
    plt.show()

# Exemplo de uso:
# df é seu DataFrame contendo as variáveis categóricas.
# plot_categorical_relationship(df, 'Sex', 'Survived')



def count_outliers(df, column):
    """
    Conta o número de outliers em uma coluna utilizando o método IQR (Intervalo Interquartil).
    
    Parameters:
        df (pd.DataFrame): DataFrame contendo os dados.
        column (str): Nome da coluna para a qual contar os outliers.
        
    Returns:
        int: Número de outliers encontrados.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()



def plot_boxplots_with_outliers(df, columns):
    """
    Plota boxplots para as colunas especificadas e exibe a contagem de outliers em cada gráfico.
    
    Parameters:
        df (pd.DataFrame): DataFrame contendo os dados.
        columns (list): Lista de colunas para plotar.
    """
    # Cria subplots para os boxplots
    fig, axes = plt.subplots(1, len(columns), figsize=(15, 5))

    # Loop pelas colunas e cria boxplots
    for ax, column in zip(axes, columns):
        sns.boxplot(data=df, x=column, ax=ax)
        outlier_count = count_outliers(df, column)
        ax.set_title(f'{column} (Outliers: {outlier_count})')

    plt.tight_layout()
    plt.show()

# Exemplo de uso:
# df_train é seu DataFrame
# columns_to_analyze = ['Age', 'SibSp', 'Parch', 'Fare']
# plot_boxplots_with_outliers(df_train, columns_to_analyze)



def plot_correlation_heatmap(df):
    """
    Plota um mapa de calor da correlação entre variáveis numéricas no DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame contendo os dados.
    """
    # Calcula a correlação entre as variáveis numéricas
    correlation_matrix = df.select_dtypes(include=['number']).corr()

    # Cria o mapa de calor
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True, cbar_kws={"shrink": .8})
    plt.title('Correlação entre Variáveis Numéricas')
    plt.show()

# Exemplo de uso:
# df_train é seu DataFrame
# plot_correlation_heatmap(df_train)


def count_outliers(df, column):
    """
    Conta o número de outliers em uma coluna utilizando o método IQR (Intervalo Interquartil).
    
    Parameters:
        df (pd.DataFrame): DataFrame contendo os dados.
        column (str): Nome da coluna para a qual contar os outliers.
        
    Returns:
        tuple: (número de outliers, limite inferior, limite superior)
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
    
    return outlier_count, lower_bound, upper_bound

def plot_boxplots_hist_kde(df, columns):
    """
    Plota boxplots, histogramas e gráficos de densidade (KDE) para as colunas especificadas,
    exibindo a contagem de outliers em cada gráfico.
    
    Parameters:
        df (pd.DataFrame): DataFrame contendo os dados.
        columns (list): Lista de colunas para plotar.
    """
    # Cria subplots para os boxplots e histogramas
    fig, axes = plt.subplots(len(columns), 2, figsize=(12, 5 * len(columns)))

    # Loop pelas colunas e cria boxplots e histogramas
    for i, column in enumerate(columns):
        # Boxplot
        sns.boxplot(data=df, x=column, ax=axes[i, 0])
        outlier_count, lower_bound, upper_bound = count_outliers(df, column)
        axes[i, 0].set_title(f'{column} (Outliers: {outlier_count})')

        # Histograma e KDE
        sns.histplot(data=df, x=column, kde=True, ax=axes[i, 1], bins=30)

        # Adiciona linhas para limites de outliers no histograma
        axes[i, 1].axvline(lower_bound, color='red', linestyle='dashed', linewidth=1.5, label='Limite Inferior')
        axes[i, 1].axvline(upper_bound, color='green', linestyle='dashed', linewidth=1.5, label='Limite Superior')
        
        axes[i, 1].set_title(f'{column} - Histograma e KDE')
        axes[i, 1].legend()

        # Adiciona anotações para os limites no histograma
        axes[i, 1].text(lower_bound, axes[i, 1].get_ylim()[1] * 0.9, 'Limite Inferior', color='red')
        axes[i, 1].text(upper_bound, axes[i, 1].get_ylim()[1] * 0.9, 'Limite Superior', color='green')

    plt.tight_layout()
    plt.show()




def plot_survival_probability(df, category, target):
    """
    Plota a probabilidade de sobrevivência por categoria.

    Parameters:
        df (pd.DataFrame): DataFrame contendo os dados.
        category (str): Nome da coluna categórica para a qual calcular a probabilidade de sobrevivência.
        target (str): Nome da coluna alvo que representa a sobrevivência (ex: 'Survived').
    """
    # Calcula a porcentagem de sobreviventes por categoria
    survival_rates = df.groupby(category)[target].mean() * 100
    survival_rates = survival_rates.reset_index()
    survival_rates.columns = [category, 'Survival Rate (%)']

    # Plota o gráfico de barras
    plt.figure(figsize=(10, 6))
    sns.barplot(data=survival_rates, x=category, y='Survival Rate (%)')
    plt.title(f'Probabilidade de Sobrevivência por {category}')
    plt.xlabel(category)
    plt.ylabel('Taxa de Sobrevivência (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


    
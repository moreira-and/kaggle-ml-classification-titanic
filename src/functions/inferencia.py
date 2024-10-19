import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2

def chi_squared_test(df, target_variable):
    """
    Realiza o teste do Qui-Quadrado para variáveis categóricas em relação à variável alvo.
    
    Parameters:
        df (pd.DataFrame): DataFrame contendo as variáveis categóricas e a variável alvo.
        target_variable (str): Nome da coluna da variável alvo categórica.
        
    Returns:
        pd.Series: P-valores para cada variável independente em relação à variável alvo.
    """
    # Cria uma cópia do DataFrame para evitar alterações no original
    df_transform = df.copy()
    
    # Inicializa o LabelEncoder
    labelencoder = LabelEncoder()
    
    # Loop por todas as colunas no DataFrame e aplica o LabelEncoder
    for col in df_transform.columns:
        df_transform[col] = labelencoder.fit_transform(df_transform[col])
    
    # Realiza o teste do Qui-Quadrado
    chi2_values, p_values = chi2(df_transform.drop(columns=[target_variable]), df_transform[target_variable])
    
    # Cria um DataFrame com os p-valores
    p_values_df = pd.Series(p_values, index=df_transform.drop(columns=[target_variable]).columns)
    
    return p_values_df




def correlation_analysis(df, var1, var2, alpha=0.05):
    """
    Calcula os coeficientes de correlação de Pearson e Spearman entre duas variáveis numéricas
    e interpreta os resultados com base no nível de significância.

    Parameters:
        df (pd.DataFrame): O DataFrame que contém os dados.
        var1 (str): Nome da primeira variável numérica.
        var2 (str): Nome da segunda variável numérica.
        alpha (float, opcional): Nível de significância para os testes. Padrão é 0.05.

    Returns:
        dict: Um dicionário contendo os coeficientes de correlação, p-valores e a interpretação.
    """
    results = {}

    # Coeficiente de Correlação de Pearson
    pearson_corr, pearson_p_value = stats.pearsonr(df[var1], df[var2])
    results['Pearson Correlation'] = pearson_corr
    results['Pearson P-value'] = pearson_p_value

    if pearson_p_value < alpha:
        results['Pearson Test'] = "Rejeita a hipótese nula: as variáveis não são independentes."
    else:
        results['Pearson Test'] = "Não se rejeita a hipótese nula: as variáveis são independentes."

    # Coeficiente de Correlação de Spearman
    spearman_corr, spearman_p_value = stats.spearmanr(df[var1], df[var2])
    results['Spearman Correlation'] = spearman_corr
    results['Spearman P-value'] = spearman_p_value

    if spearman_p_value < alpha:
        results['Spearman Test'] = "Rejeita a hipótese nula: as variáveis não são independentes."
    else:
        results['Spearman Test'] = "Não se rejeita a hipótese nula: as variáveis são independentes."

    return results


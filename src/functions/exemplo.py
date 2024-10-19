def welcome_message(simulate_error: bool = False) -> str:
    """
    Retorna uma mensagem de boas-vindas para o portfólio do André (https://github.com/moreira-and).

    Esta função aceita um parâmetro booleano que determina se um erro
    deve ser simulado. Se simulate_error for True, a função gerará um erro
    de divisão por zero.

    Parâmetros:
        simulate_error (bool): Se True, simula um erro. Se False, retorna
                               a mensagem de boas-vindas.

    Retorno:
        str: A mensagem de boas-vindas ou uma mensagem de erro.
    """
    try:
        if simulate_error:
            # Simulando uma operação que pode causar um erro
            result = 1 / 0  # Isso vai gerar um erro de divisão por zero
        else:
            return "Seja bem-vindo ao portfólio do André (https://github.com/moreira-and)"

    except ZeroDivisionError as e:
        return f"Ocorreu um erro: {e}. A mensagem não pôde ser gerada."


# Exemplo de uso da função
if __name__ == "__main__":
    # Teste sem erro
    print(welcome_message(simulate_error=False))
    
    # Teste com erro
    print(welcome_message(simulate_error=True))

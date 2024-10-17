## Importação de bibliotecas
import sys
import os

# Adiciona o diretório src ao PYTHONPATH
sys.path.append('../')
from src.functions.function_example import welcome_message

class Curriculo:
    """
    Classe que representa um currículo profissional.

    A classe encapsula todas as informações relevantes para o currículo de um
    profissional, incluindo dados pessoais, perfil profissional, experiência,
    formação acadêmica, projetos, certificações, idiomas e outras qualificações.
    """

    def __init__(self, nome: str, endereco: str, telefone: str, email: str,
                 perfil_profissional: str, experiencia: str, formacao: str,
                 projetos: str, certificacoes: str, idiomas: str,
                 outras_qualificacoes: str):
        """
        Inicializa uma nova instância da classe Curriculo.

        Parâmetros:
            nome (str): Nome do profissional.
            endereco (str): Endereço do profissional.
            telefone (str): Número de telefone para contato.
            email (str): Endereço de e-mail do profissional.
            perfil_profissional (str): Descrição do perfil profissional.
            experiencia (str): Experiência profissional detalhada.
            formacao (str): Formação acadêmica do profissional.
            projetos (str): Projetos acadêmicos ou práticos realizados.
            certificacoes (str): Certificações obtidas.
            idiomas (str): Idiomas que o profissional possui conhecimento.
            outras_qualificacoes (str): Outras qualificações relevantes.
        """
        self.nome = nome
        self.endereco = endereco
        self.telefone = telefone
        self.email = email
        self.perfil_profissional = perfil_profissional
        self.experiencia = experiencia
        self.formacao = formacao
        self.projetos = projetos
        self.certificacoes = certificacoes
        self.idiomas = idiomas
        self.outras_qualificacoes = outras_qualificacoes
        pass

    def exibir_curriculo(self) -> None:
        """
        Exibe as informações do currículo formatadas de forma legível.
        """
        curriculo_info = f"""
        {self.nome}
        {self.endereco}
        {self.telefone} ∙ {self.email}
        
        PERFIL PROFISSIONAL
        {self.perfil_profissional}
        
        EXPERIÊNCIA PROFISSIONAL
        {self.experiencia}
        
        FORMAÇÃO ACADÊMICA
        {self.formacao}
        
        PROJETOS ACADÊMICOS
        {self.projetos}
        
        CERTIFICAÇÕES
        {self.certificacoes}
        
        IDIOMAS
        {self.idiomas}
        
        OUTRAS QUALIFICAÇÕES
        {self.outras_qualificacoes}
        """
        print(curriculo_info)

    def show_welcome_message(self, simulate_error: bool = False) -> str:
        """
        Chama a função welcome_message importada e retorna a mensagem.

        Parâmetros:
            simulate_error (bool): Se True, simula um erro. Se False, chama
                                    a função normal.

        Retorno:
            str: A mensagem de boas-vindas ou uma mensagem de erro.
        """
        return welcome_message(simulate_error)  # Chama a função importada


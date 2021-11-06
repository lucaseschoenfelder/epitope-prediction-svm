from classes.features import AAP, AAT
from classes.command_line import Cli
from classes.file import File
from classes.model import Model
from utils.setup_logger import logger

from time import time
from os.path import exists

import sys
import numpy as np


# Definição para inicializar a Interface de linha de comando
cli = Cli()

# Definição para iniciar o manipulador de arquivos
file_handler = File()

# Defninição para iniciar a classe que irá definir o modelo
model = Model()


if __name__=='__main__':
    logger.info("Iniciando aplicação")

    time_init = time()

    # Buscando o caminho completo dos arquivos de peptídeos passado como parâmetro
    positive_file = cli.get_arg_from_cli('positive_file')
    negative_file = cli.get_arg_from_cli('negative_file')

    # Nome do dataset para salvar os dicionários de antigenicidade
    dataset_name = cli.get_arg_from_cli('dataset_name')

    # Realização da leitura dos arquivos com as sequências de peptídeos
    positive_sequences = file_handler.get_sequences_from_file(positive_file)
    negative_sequences = file_handler.get_sequences_from_file(negative_file)

    # Remoção das variáveis que apontam para os arquivos de sequência
    # A partir desse ponto do código, apenas as variáveis com as sequências serão utilizadas
    del positive_file
    del negative_file

    feature_list = []

    dataset = positive_sequences + negative_sequences
    target = [1] * len(positive_sequences) + [0] * len(negative_sequences)

    features = np.empty([len(dataset), 1])

    if cli.get_arg_from_cli('aat_feature'):

        # Guarda o path completo para o arquivo
        path_to_file = f"scale/{dataset_name}/aat_scale.txt"
            
        # Inicializa a classe AAT
        aat = AAT()
        
        # Testa se o arquivo já existe, se sim, abre o arquivo, se não, salva o arquivo para que não precise ser gerado novamente se necessário
        if not exists(path_to_file):

            logger.info(f"Dicionário AAT para o dataset {dataset_name} não existe!")

            # Realiza a criação dos arquivos com a escala AAT para as sequências passadas como parâmetro
            aat_scale = aat.generate_aat_scale(positive_sequences, negative_sequences)

            # Salva escala gerada em um arquivo
            file_handler.save_dict_in_file(path_to_file, aat_scale)
        else:

            logger.info(f"Dicionário AAT para o dataset {dataset_name} já existe, será lido do arquivo!")

            # Caso o arquivo já exista é realizada a leitura da escala já calculada
            aat_scale = file_handler.transform_file_in_dict(path_to_file)

        # Extração da feature aat para cada peptídeo do dataset
        aat_feature = aat.extract_aat_feature(dataset, aat_scale)

        features = np.column_stack((features, aat_feature))

        # Salvando a feature que será utilizada para treinar o modelo
        feature_list.append('aat')

    if cli.get_arg_from_cli('aap_feature'):

        # Guarda o path completo para o arquivo
        path_to_file = f"scale/{dataset_name}/aap_scale.txt"

        # Inicializa a classe AAP
        aap = AAP()

        # Testa se o arquivo já existe, se sim, abre o arquivo, se não, salva o arquivo para que não precise ser gerado novamente se necessário
        if not exists(path_to_file):

            logger.info(f"Dicionário AAP para o dataset {dataset_name} não existe!")

            # Realiza a criação dos arquivos com a escala AAP para as sequências passadas como parâmetro
            aap_scale = aap.generate_aap_scale(positive_sequences, negative_sequences)

            # Salva a escala gerada em um arquivo
            file_handler.save_dict_in_file(path_to_file, aap_scale)
        else:

            logger.info(f"Dicionário AAP para o dataset {dataset_name} já existe, será lido do arquivo!")
            
            # Caso o arquivo já exista é realizada a leitura da escala já calculada
            aap_scale = file_handler.transform_file_in_dict(path_to_file)

        # Extração da feature aap para cada peptídeo do dataset
        aap_feature = aap.extract_aap_feature(dataset, aap_scale)

        features = np.column_stack((features, aap_feature))

        # Salvando a feature que será utilizada para treinar o modelo
        feature_list.append('aap')

    if not len(feature_list):
        logger.error("Não foi selecionada nenhuma feature para treinamento do modelo. Programa será encerrado!")
        sys.exit()

    logger.info(f"Features que estão sendo utilizadas para treinar o modelo: {feature_list}")

    x, y = model.prepare_x_and_y(features, target)

    grid_search = model.grid_search(x, target)

    logger.info(grid_search.best_score_)
    logger.info(grid_search.best_params_)

    time_end = time()

    logger.debug(f"Tempo gasto em segundos para executar toda a aplicação: {time_end - time_init} segundos")
    logger.info("Aplicação finalizada")

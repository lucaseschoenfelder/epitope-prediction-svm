from classes.features import AAP, AAT, AAC, DPC, CTD, ProtVec
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

    positive_sequences_validation = file_handler.get_sequences_from_file("dataset/independent/independent-pos.txt")
    negative_sequences_validation = file_handler.get_sequences_from_file("dataset/independent/independent-neg.txt")

    # Remoção das variáveis que apontam para os arquivos de sequência
    # A partir desse ponto do código, apenas as variáveis com as sequências serão utilizadas
    del positive_file
    del negative_file

    feature_list = []

    dataset = positive_sequences + negative_sequences
    validation_dataset = positive_sequences_validation + negative_sequences_validation
    
    target = [1] * len(positive_sequences) + [0] * len(negative_sequences)
    validation_target = [1] * len(positive_sequences_validation) + [0] * len(negative_sequences_validation)

    features = np.empty([len(dataset), 1])
    validation_features = np.empty([len(validation_dataset), 1])

    if cli.get_arg_from_cli('aat_feature'):

        # Guarda o path completo para o arquivo
        path_to_file = f"scale/{dataset_name}/aat_scale.txt"
            
        # Inicializa a classe AAT
        aat = AAT()
        
        # Testa se o arquivo já existe, se sim, abre o arquivo, se não, salva o arquivo para que não precise ser gerado novamente se necessário
        if not exists(path_to_file):

            logger.info(f"Dicionário AAT para o dataset {dataset_name} não existe!")

            # Realiza a criação dos arquivos com a escala AAT para as sequências passadas como parâmetro
            aat_scale = aat.generate_aat_scale(positive_sequences_validation, negative_sequences_validation)

            # Salva escala gerada em um arquivo
            file_handler.save_dict_in_file(path_to_file, aat_scale)
        else:

            logger.info(f"Dicionário AAT para o dataset {dataset_name} já existe, será lido do arquivo!")

            # Caso o arquivo já exista é realizada a leitura da escala já calculada
            aat_scale = file_handler.transform_file_in_dict(path_to_file)

        validation_aat_scale = aat.generate_aat_scale(positive_sequences, negative_sequences)

        # Extração da feature aat para cada peptídeo do dataset
        aat_feature = aat.extract_aat_feature(dataset, aat_scale)
        validation_aat_feature = aat.extract_aat_feature(validation_dataset, validation_aat_scale)

        features = np.column_stack((features, aat_feature))
        validation_features = np.column_stack((validation_features, validation_aat_feature))
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

        validation_aap_scale = aap.generate_aap_scale(positive_sequences_validation, negative_sequences_validation)

        # Extração da feature aap para cada peptídeo do dataset
        aap_feature = aap.extract_aap_feature(dataset, aap_scale)
        validation_aap_feature = aap.extract_aap_feature(validation_dataset, validation_aap_scale)

        features = np.column_stack((features, aap_feature))
        validation_features = np.column_stack((validation_features, validation_aap_feature))

        # Salvando a feature que será utilizada para treinar o modelo
        feature_list.append('aap')

    if cli.get_arg_from_cli('aac_feature'):

        # Inicializa a classe AAC
        aac = AAC()

        # Extração da feature aac para cada peptídeo do dataset
        aac_feature = aac.extract_aac_feature(dataset)

        validation_aac_feature = aac.extract_aac_feature(validation_dataset)

        features = np.column_stack((features, aac_feature))
        validation_features = np.column_stack((validation_features, validation_aac_feature))

        # Salvando a feature que será utilizada para treinar o modelo
        feature_list.append('aac')

    if cli.get_arg_from_cli('ctd_feature'):

        # Inicializa a classe DPC
        ctd = CTD()

        # Extração da feature aac para cada peptídeo do dataset
        ctd_feature = ctd.extract_ctd_feature(dataset)
        validation_ctd_feature = ctd.extract_ctd_feature(validation_dataset)

        features = np.column_stack((features, ctd_feature))
        validation_features = np.column_stack((validation_features, validation_ctd_feature))

        # Salvando a feature que será utilizada para treinar o modelo
        feature_list.append('ctd')

    if cli.get_arg_from_cli('dpc_feature'):

        # Inicializa a classe AAC
        dpc = DPC()

        # Extração da feature aac para cada peptídeo do dataset
        dpc_feature = dpc.extract_dpc_feature(dataset)
        
        features = np.column_stack((features, dpc_feature))

        # Salvando a feature que será utilizada para treinar o modelo
        feature_list.append('dpc')
    
    
    if cli.get_arg_from_cli('protvec_feature'):

        # Inicializa a classe AAC
        protvec = ProtVec()

        # Extração da feature aac para cada peptídeo do dataset
        protvec_feature = protvec.extract_protvec_feature(dataset, "./protvec/sp_sequences_4mers_vec.txt")

        features = np.column_stack((features, protvec_feature))

        # Salvando a feature que será utilizada para treinar o modelo
        feature_list.append('protvec')

    if not len(feature_list):
        logger.error("Não foi selecionada nenhuma feature para treinamento do modelo. Programa será encerrado!")
        sys.exit()

    logger.info(f"Features que estão sendo utilizadas para treinar o modelo: {feature_list}")

    x, y = model.prepare_x_and_y(features, target)
    x_val, y_val = model.prepare_x_and_y(validation_features, validation_target)
    
    logger.info(f"Quantidade de features por peptídeo: {len(x[0])}")

    if cli.get_arg_from_cli('model'):
        logger.info(f'Será testado apenas o modelo {cli.get_arg_from_cli("model")}')
        #estimators, best_params_per_estimator = model.grid_search_models(x, y, cli.get_arg_from_cli('params'), cli.get_arg_from_cli('result_path'), [cli.get_arg_from_cli('model')])    
        gridsearch_model = model.grid_search(x, y, cli.get_arg_from_cli('result_path'), cli.get_arg_from_cli('model'))

        results = gridsearch_model.cv_results_
        bi = gridsearch_model.best_index_

        logger.info(  f"Melhores resultados: \n \
                        roc_auc: {results['mean_test_auc_score'][bi]},\n \
                        accuracy: {results['mean_test_accuracy'][bi]},\n  \
                        precision_+:{results['mean_test_scores_p_1'][bi]},\n \
                        recall_+:{results['mean_test_scores_r_1'][bi]},\n \
                        f1_+:{results['mean_test_scores_f_1_1'][bi]},\n \
                        precision_-:{results['mean_test_scores_p_0'][bi]},\n \
                        recall_-:{results['mean_test_scores_r_0'][bi]},\n \
                        f1_-:{results['mean_test_scores_f_1_0'][bi]},\n \
                        precision_micro:{results['mean_test_precision_micro'][bi]},\n \
                        precision_macro:{results['mean_test_precision_macro'][bi]},\n \
                        mcc:{results['mean_test_mcc'][bi]}")

        logger.info(f"Melhor score: {gridsearch_model.best_score_}")
        logger.info(f"Melhores parâmetros: {gridsearch_model.best_params_}")

        #model.plot_grid_search(gridsearch_model)
        model.plot_search_results(gridsearch_model, cli.get_arg_from_cli("dataset_name"))
        #model.table_grid_search(gridsearch_model, all_ranks=True)

        # logger.info(f'Dataset de validação:')
        # logger.info(f"Quantidade de features por peptídeo (validacao): {len(x_val[0])}")


        # preds = gridsearch_model.predict(x_val)

        # logger.info(  f"Resultados validação: \n \
        #                 roc_auc: {roc_auc_score(y_val, preds)},\n \
        #                 accuracy: {accuracy_score(y_val, preds)},\n  \
        #                 precision:{precision_score(y_val, preds)},\n \
        #                 recall:{recall_score(y_val, preds)},\n \
        #                 mcc:{matthews_corrcoef(y_val, preds)}")

        time_end = time()

        logger.debug(f"Tempo gasto em segundos para executar toda a aplicação: {time_end - time_init} segundos")
        logger.info("Aplicação finalizada")        
    elif cli.get_arg_from_cli('params'):
        # Carrega os params e roda ensemble com todos os modelos definidos pela variavel estimators
        estimators, best_params_per_estimator = model.get_models_and_params(x, y, cli.get_arg_from_cli('params'), cli.get_arg_from_cli('result_path'))

        #caso queira rodar o ensemble de todas as possiveis permutacoes de combinações de modelos, rodar o programa com o argumento --combine
        best_ensemble_model = model.grid_search_ensemble(x, y, estimators, best_params_per_estimator, cli.get_arg_from_cli('result_path'), cli.get_arg_from_cli('combine'))
        
        logger.info(  f"Melhores resultados: \n \
                        roc_auc: {best_ensemble_model[1]['mean_test_auc_score']},\n \
                        accuracy: {best_ensemble_model[1]['mean_test_accuracy']},\n  \
                        precision_+:{best_ensemble_model[1]['mean_test_scores_p_1']},\n \
                        recall_+:{best_ensemble_model[1]['mean_test_scores_r_1']},\n \
                        f1_+:{best_ensemble_model[1]['mean_test_scores_f_1_1']},\n \
                        precision_-:{best_ensemble_model[1]['mean_test_scores_p_0']},\n \
                        recall_-:{best_ensemble_model[1]['mean_test_scores_r_0']},\n \
                        f1_-:{best_ensemble_model[1]['mean_test_scores_f_1_0']},\n \
                        precision_micro:{best_ensemble_model[1]['mean_test_precision_micro']},\n \
                        precision_macro:{best_ensemble_model[1]['mean_test_precision_macro']},\n \
                        mcc:{best_ensemble_model[1]['mean_test_mcc']}")

        logger.info(f"Melhor score: {best_ensemble_model[1]['mean_test_mcc']}")
        logger.info(f"Melhor combinação: {best_ensemble_model[0]}")

        time_end = time()

        logger.debug(f"Tempo gasto em segundos para executar toda a aplicação: {time_end - time_init} segundos")
        logger.info("Aplicação finalizada")
    # else:
    #     logger.info(f'Rodando ensemble com todos os parametros default')
    #     gridsearch_model = model.cross_validate_ensemble(None, None, x, y, cli.get_arg_from_cli('result_path'))

    #     results = gridsearch_model.cv_results_
    #     bi = gridsearch_model.best_index_

    #     logger.info(  f"Resultados encontrados: \n \
    #                     roc_auc: {results['mean_test_auc_score'][bi]},\n \
    #                     accuracy: {results['mean_test_accuracy'][bi]},\n  \
    #                     precision +:{results['mean_test_scores_p_1'][bi]},\n \
    #                     recall +:{results['mean_test_scores_r_1'][bi]},\n \
    #                     f1 +:{results['mean_test_scores_f_1_1'][bi]},\n \
    #                     precision -:{results['mean_test_scores_p_0'][bi]},\n \
    #                     recall -:{results['mean_test_scores_r_0'][bi]},\n \
    #                     f1 -:{results['mean_test_scores_f_1_0'][bi]},\n \
    #                     precision_micro:{results['mean_test_precision_micro'][bi]},\n \
    #                     f1 -:{results['mean_test_precision_macro'][bi]},\n \
    #                     mcc -:{results['mean_test_mcc'][bi]}")

    #     logger.info(f"Score obtido: {gridsearch_model.best_score_}")
    #     # logger.info(f"Melhores parâmetros: {gridsearch_model.best_params_}")

    #     time_end = time()

    #     logger.debug(f"Tempo gasto em segundos para executar toda a aplicação: {time_end - time_init} segundos")
    #     logger.info("Aplicação finalizada") 

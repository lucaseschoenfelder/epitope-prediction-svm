from genericpath import exists
from pydoc import cli
from numpy import ndarray
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics._scorer import make_scorer
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
# from sklearn.tree import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from utils.setup_logger import logger
from itertools import combinations
from time import time
import json
import os.path  
import numpy as np
import pandas as pd
# from pip._internal.operations import freeze
# x = freeze.freeze()
# for p in x:
#     print(p)
# input()


class Model():
    """Classe que contém as definições do modelo de ML que será usado no projeto"""

    def __init__(self) -> None:
        pass
    
    def precision_0(y_true, y_pred, labels=None, average='binary', sample_weight=None):
        '''
        :param y_true:
        :param y_pred:
        :param labels:
        :param average:
        :param sample_weight:
        :return: calculate prec for neg class
        '''
        p, _, _, _ = precision_recall_fscore_support(y_true, y_pred,
                                                     beta=1,
                                                     labels=labels,
                                                     pos_label=0,
                                                     average=average,
                                                     warn_for=('f-score',),
                                                     sample_weight=sample_weight)
        return p

    def recall_0(y_true, y_pred, labels=None, average='binary', sample_weight=None):
        '''
        :param y_true:
        :param y_pred:
        :param labels:
        :param average:
        :param sample_weight:
        :return: calculate recall for neg class
        '''
        _, r, _, _ = precision_recall_fscore_support(y_true, y_pred,
                                                     beta=1,
                                                     labels=labels,
                                                     pos_label=0,
                                                     average=average,
                                                     warn_for=('f-score',),
                                                     sample_weight=sample_weight)
        return r

    def f1_0(y_true, y_pred, labels=None, average='binary', sample_weight=None):
        '''
        :param y_true:
        :param y_pred:
        :param labels:
        :param average:
        :param sample_weight:
        :return: calculate f1 for neg class
        '''
        _, _, f, _ = precision_recall_fscore_support(y_true, y_pred,
                                                     beta=1,
                                                     labels=labels,
                                                     pos_label=0,
                                                     average=average,
                                                     warn_for=('f-score',),
                                                     sample_weight=sample_weight)
        return f

    def cross_validate_ensemble(self, estimators, params, x, y, path_csv_result):
        logger.info(f"Iniciando Cross Validate para o modelo ensemble")

        time_init = time()

        cross_valid = StratifiedKFold(n_splits=5)

        scoring = { 
            'auc_score': 'roc_auc',
            'accuracy': 'accuracy',
            'scores_p_1': 'precision',
            'scores_r_1': 'recall',
            'scores_f_1_1': 'f1',
            'scores_p_0': make_scorer(Model.precision_0),
            'scores_r_0': make_scorer(Model.recall_0),
            'scores_f_1_0': make_scorer(Model.f1_0),
            'mcc': make_scorer(matthews_corrcoef),
            'precision_micro': 'precision_micro',
            'precision_macro': 'precision_macro', 
            'recall_macro': 'recall_macro',
            'recall_micro': 'recall_micro', 
            'f1_macro': 'f1_macro', 
            'f1_micro': 'f1_micro'
        }
        
        if estimators == None:
            logger.info(f'Entrei no cross_validate_ensemble sem estimadores, vou utilizar todos disponíveis')
            estimators = []
            logger.info('Ensemble testará RandomForestClassifier')
            estimators.append(('rfc', RandomForestClassifier()))
            logger.info('Ensemble testará AdaBoostClassifier')
            estimators.append(('abc', AdaBoostClassifier()))
            logger.info('Ensemble testará GradientBoostingClassifier')
            estimators.append(('gbc', GradientBoostingClassifier()))
            logger.info('Ensemble testará MLPClassifier')
            estimators.append(('mlp', MLPClassifier()))
            logger.info('Ensemble testará ExtraTreesClassifier')
            estimators.append(('etc', ExtraTreesClassifier()))
            logger.info('Ensemble testará SVC')
            estimators.append(('svc', SVC(probability=True)))
            
            params = dict()
            for estimator in estimators:
                for k, v in estimator[1].get_params().items():
                    # logger.info(f'{estimator[0]}__{k} = {v}')
                    params[f'{estimator[0]}__{k}'] = [v] 
            logger.info(f'params: {params}')

        eclf = VotingClassifier(estimators, voting = 'soft', verbose = True)

        # if params == None or len(params)==0:
        #     logger.info(f'Usando GridSearch com parametros default')
        #     logger.info(f'Estimators={estimators}')
        #     grid_search = GridSearchCV(estimator=eclf,
        #                             param_grid=params,
        #                             scoring=scoring, 
        #                             cv=cross_valid,
        #                             refit='auc_score',
        #                             n_jobs=40, 
        #                             verbose=2)
        # else:
        grid_search = GridSearchCV(estimator=eclf,
                                param_grid=params,
                                scoring=scoring, 
                                cv=cross_valid,
                                refit='mcc',
                                n_jobs=40, 
                                verbose=2)

        grid_search.fit(x, y)

        logger.info(f"Resultados obtidos em todos os treinamentos: {grid_search.cv_results_}")

        results_dataframe = pd.DataFrame(data=grid_search.cv_results_)

        results_dataframe.to_csv(path_csv_result)

        time_end = time()

        logger.debug(f"Tempo gasto em segundos para executar o GridSearchCV: {time_end - time_init} segundos")
        logger.info("FInalizado GridSearchCV para o modelo")
        
        return grid_search    

    def grid_search_ensemble(self, x, y, estimators, best_params_per_model, path_csv_result=None, testAllCombinations=False):
        logger.info(f"Iniciando GridSearchCV para o modelo ensemble. estimators = {estimators}. best_params_per_model = {best_params_per_model}")
        # estimators = [('abc', AdaboostClassifier())]

        time_init = time()

        # best_params_per_model = {
        #     'rfc' : {
        #         'rfc__max_features': ['sqrt'], 'rfc__min_samples_split': [10], 'rfc__n_estimators': [860]
        #     },
        #     'abc': {
        #         'abc__n_estimators': [60]
        #     },
        #     'gbc' : {
        #         'gbc__max_features': [None], 'gbc__min_samples_split': [7], 'gbc__n_estimators': [100]
        #     },
        #     'mlp' : {
        #         'mlp__activation': ['logistic'], 'mlp__learning_rate': ['adaptive'], 'mlp__max_iter': [1000], 'mlp__solver': ['adam']
        #     },
        #     'etc' : {
        #         'etc__max_features': [None], 'etc__min_samples_split': [9], 'etc__splitter': ['random']
        #     },
        #     'svc' : {
        #         'svc__C': [500], 'svc__gamma': [0.0001], 'svc__kernel': ['rbf']
        #     }
        # }

        
        if testAllCombinations:
            estimators_combinations = list()
            logger.info(f'Gerando lista de combinações de classificadores')
            for n in range(2, len(estimators) + 1):
                estimators_combinations += list(combinations(estimators, n))
            
            if len(estimators_combinations) < 1:
                logger.warn(f'Cancelando execução pois nenhuma combinação foi gerada.')
                exit(-1)

            for combination in estimators_combinations:
                logger.info(f'combination : {combination}')
        else:
            estimators_combinations = [estimators]

        logger.info(f'estimators_combinations = {estimators_combinations}')
        results = dict()
        for combination in estimators_combinations:
            params = dict()
            path_to_ensemble_combination = f'{path_csv_result}_ensemble'
            for model_tuple in combination:
                logger.info(f'model_tuple = {model_tuple}')
                params.update(best_params_per_model[model_tuple[0]])
                path_to_ensemble_combination += f'_{model_tuple[0]}'
            
            logger.info(f'params = {params}')
            cv_ensemble = self.cross_validate_ensemble(combination, params, x, y, path_to_ensemble_combination)
            
            combination_key = path_to_ensemble_combination.split("ensemble_")[1]

            logger.info(f'Modelo {combination_key} obteve score {cv_ensemble.best_score_}')

            results[combination_key] = dict()

            ensemble_results = cv_ensemble.cv_results_
            bi = cv_ensemble.best_index_

            results[combination_key]["mean_test_auc_score"] = ensemble_results['mean_test_auc_score'][bi]
            results[combination_key]["mean_test_accuracy"] = ensemble_results['mean_test_accuracy'][bi]
            results[combination_key]["mean_test_scores_p_1"] = ensemble_results['mean_test_scores_p_1'][bi]
            results[combination_key]["mean_test_scores_r_1"] = ensemble_results['mean_test_scores_r_1'][bi]
            results[combination_key]["mean_test_scores_f_1_1"] = ensemble_results['mean_test_scores_f_1_1'][bi]
            results[combination_key]["mean_test_scores_p_0"] = ensemble_results['mean_test_scores_p_0'][bi]
            results[combination_key]["mean_test_scores_r_0"] = ensemble_results['mean_test_scores_r_0'][bi]
            results[combination_key]["mean_test_scores_f_1_0"] = ensemble_results['mean_test_scores_f_1_0'][bi]
            results[combination_key]["mean_test_precision_micro"] = ensemble_results['mean_test_precision_micro'][bi]
            results[combination_key]["mean_test_precision_macro"] = ensemble_results['mean_test_precision_macro'][bi]
            results[combination_key]["mean_test_mcc"] = ensemble_results['mean_test_mcc'][bi]

            del params            
        
        if testAllCombinations:
            logger.info(f'Fim do gridsearch ensemble. Resultados:')
            for key, val in results.items():
                logger.info(f'{key} : {val}')
        
        best_ensemble_model = max(results, key=lambda x : results[x]["mean_test_mcc"])
        return (best_ensemble_model, results[best_ensemble_model])

    def grid_search(self, x, y, path_csv_result=None, model_param="SVC"):
        """Define o gridsearch para ser utilizado para valorar os melhores parâmetros para o modelo passado como parâmetro

            Utilizando o parâmetro path_csv_result o resultado completo do GridSearchCV será salvo no path fornecido
        """

        logger.info(f"Iniciando GridSearchCV para o modelo {model_param}")

        time_init = time()

        cross_valid = StratifiedKFold(n_splits=5)

        scoring = { 
            'auc_score': 'roc_auc',
            'accuracy': 'accuracy',
            'scores_p_1': 'precision',
            'scores_r_1': 'recall',
            'scores_f_1_1': 'f1',
            'scores_p_0': make_scorer(Model.precision_0),
            'scores_r_0': make_scorer(Model.recall_0),
            'scores_f_1_0': make_scorer(Model.f1_0),
            'mcc': make_scorer(matthews_corrcoef),
            'precision_micro': 'precision_micro',
            'precision_macro': 'precision_macro', 
            'recall_macro': 'recall_macro',
            'recall_micro': 'recall_micro', 
            'f1_macro': 'f1_macro', 
            'f1_micro': 'f1_micro'
        }

        if model_param == "rfc":
            logger.info(f'Usando modelo Random Forest')
            model = RandomForestClassifier()
            params = { 
                'n_estimators': [i for i in range(20, 1001, 20)],
                'max_features': ['sqrt', None],
                'min_samples_split' : [i for i in range(2, 11, 1)]
            }
        elif model_param == "abc":
            logger.info(f'Usando modelo AdaBoost')
            model = AdaBoostClassifier()
            params = {
                'n_estimators' : [i for i in range(50, 1001, 50)],
                'algorithm' : ['SAMME', 'SAMME.R'],
                'learning_rate' : [1, 0.5, 0.25, 0.1, 0.05, 0.01]
            }
        elif model_param == "gbc":
            logger.info(f'Usando modelo GradientBoost')
            model = GradientBoostingClassifier()
            params = {
                'n_estimators' : [5, 10, 20, 50, 100, 200, 250, 500],
                'max_features': ['sqrt', None],
                'learning_rate' : [1, 0.5, 0.25, 0.1, 0.05, 0.01],
                'subsample' : [0.5, 0.6, 0.7, 0.8, 0.85, 0.9],
                'max_depth': [3, 5, 7, 9]
            }
        elif model_param == "mlp":
            logger.info(f'Usando modelo Multilayer Perceptron')
            model = MLPClassifier()
            params = {
                'max_iter' : [500],
                'activation' : ['logistic', 'tanh', 'relu'],
                'solver' : ['lbfgs', 'adam'],
                'learning_rate': ['constant','adaptive']
            }
        elif model_param == "etc":
            logger.info(f'Usando modelo Extremely Randomized Tree')
            model = ExtraTreesClassifier()
            params = {
                'n_estimators' : [100, 200, 500],
                'max_depth': [1, 2, 5, 10, 20, 50, None] ,
                'max_features': ['sqrt', None],
                'min_samples_split' : [i for i in range(2, 11, 1)]
            }
        else:
            logger.info(f'Usando modelo SVC')
            model = SVC(probability=True)
            params = {  
                'kernel': ['rbf'],
                'C': [500, 250],
                'gamma': [0.001, 0.0001]
            }    
        
        logger.info(f'Params usados: {params}')

        grid_search = GridSearchCV(estimator=model,
                                   param_grid=params,
                                   scoring=scoring, 
                                   cv=cross_valid,
                                   refit='mcc',
                                   n_jobs=40, 
                                   verbose=2)

    
        grid_search.fit(x, y)

        logger.info(f"Resultados obtidos em todos os treinamentos: {grid_search.cv_results_}")

        results_dataframe = pd.DataFrame(data=grid_search.cv_results_)

        results_dataframe.to_csv(path_csv_result)

        time_end = time()

        logger.debug(f"Tempo gasto em segundos para executar o GridSearchCV: {time_end - time_init} segundos")
        logger.info("FInalizado GridSearchCV para o modelo {model_param}")
        
        return grid_search

    def get_models_and_params(self, x, y, path_to_params_file, path_csv_result=None,
                            models_to_evaluate=['abc', 'gbc', 'etc']):

        logger.info(f'Entrei no get_models_and_params com models_to_evaluate = {models_to_evaluate} e path_to_params_file={path_to_params_file}')
        # models_to_evaluate = [
        #     'abc', 'rfc', 'gbc', 'mlp', 'etc', 'svc'
        # ]

        
        path_to_params_json = f'/../model_params_dict/{path_to_params_file}'
        # if not exists(f'{os.path.dirname(__file__)}{path_to_params_json}'):
        #     logger.info("Arquivo com os melhores parametros não encontrado. Iniciando busca pelos melhores parametros de cada modelo")
        #     best_params_per_estimator = dict()
        #     for model in models_to_evaluate:

        #         grid_search = self.grid_search(x, y, path_csv_result, model)
        #         logger.info(f'Melhores parametros para o modelo {model}: {grid_search.best_params_}')
        #         best_params = grid_search.best_params_

        #         best_params_per_estimator[model] = dict()

        #         for param, best_value in best_params.items():
        #             key_name = f'{model}__{param}'
        #             best_params_per_estimator[model][key_name] = [best_value]

        #     with open(f'{os.path.dirname(__file__)}{path_to_params_json}', 'w') as fp:
        #         logger.info(f'Salvando os melhores parametros em arquivo...')
        #         json.dump(best_params_per_estimator, fp, indent=4)     
        # else:
        #     logger.info("Arquivo com os melhores parametros encontrado! Lendo arquivo...")
        #     with open(f'{os.path.dirname(__file__)}{path_to_params_json}', 'r') as fp:
        #         best_params_per_estimator = json.load(fp)
        if not exists(f'{os.path.dirname(__file__)}{path_to_params_json}'):
            logger.info(f'Arquivo com os melhors parametros não encontrado. Abortando execução...')
            exit(0)
        else:
            logger.info("Arquivo com os melhores parametros encontrado! Lendo arquivo...")
            with open(f'{os.path.dirname(__file__)}{path_to_params_json}', 'r') as fp:
                best_params_per_estimator = json.load(fp)
            
        logger.info(f'best_params_per_estimator = {best_params_per_estimator}')

        estimators = []
        
        for model in best_params_per_estimator.keys():
            if model == 'rfc':
                logger.info('Ensemble testará RandomForestClassifier')
                estimators.append(('rfc', RandomForestClassifier()))
            elif model == 'abc':
                logger.info('Ensemble testará AdaBoostClassifier')
                estimators.append(('abc', AdaBoostClassifier()))
            elif model == 'gbc':
                logger.info('Ensemble testará GradientBoostingClassifier')
                estimators.append(('gbc', GradientBoostingClassifier()))
            elif model == 'mlp':
                logger.info('Ensemble testará MLPClassifier')
                estimators.append(('mlp', MLPClassifier()))
            elif model == 'etc':
                logger.info('Ensemble testará ExtraTreesClassifier')
                estimators.append(('etc', ExtraTreesClassifier()))
            elif model == 'svc':
                logger.info('Ensemble testará SVC')
                estimators.append(('svc', SVC(probability=True)))
            else:
                logger.warn("Modelo desconhecido")

        logger.info(f'Estimators a serem testados: {estimators}')

        try:
            best_params_per_estimator
        except NameError:
            return estimators, dict()
        else:
            return estimators, best_params_per_estimator

    def prepare_x_and_y(self, features: ndarray, target: list):
        """Método que irá transformar as features e os rótulos em objetos que podem ser interpretados pelo GridSearchCV"""
    
        logger.info("Iniciando preparação das features e dos rótulos passados como parâmetro")

        time_init = time()

        scaling = StandardScaler()
        scaling.fit(features[:,1:])
        x = scaling.transform(features[:,1:])

        y = np.array(target)

        time_end = time()

        logger.debug(f"Tempo gasto em segundos para realizar as tratativas das features e dos rótulos: {time_end - time_init} segundos")
        logger.info("Finalizada a preparação das features e dos rótulos")


        return x, y

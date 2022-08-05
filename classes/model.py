from numpy import ndarray
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics._scorer import make_scorer
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.neural_network import MLPClassifier
from utils.setup_logger import logger
from itertools import combinations
from time import time

import numpy as np
import pandas as pd

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
        
        eclf = VotingClassifier(estimators, voting = 'soft', verbose = True)

        grid_search = GridSearchCV(estimator=eclf,
                                   param_grid=params,
                                   scoring=scoring, 
                                   cv=cross_valid,
                                   refit='auc_score',
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

    def grid_search_ensemble(self, x, y, path_csv_result=None):
        logger.info(f"Iniciando GridSearchCV para o modelo ensemble")

        time_init = time()

        rfc = RandomForestClassifier()
        abc = AdaBoostClassifier()
        gbc = GradientBoostingClassifier()
        mlp = MLPClassifier()
        etc = ExtraTreeClassifier()
        svc = SVC(probability=True)
        estimators = [('rfc', rfc), ('abc', abc), ('gbc', gbc), ('mlp', mlp), ('etc', etc), ('svc', svc)]

        best_params_per_model = {
            'rfc' : {
                'rfc__max_features': ['sqrt'], 'rfc__min_samples_split': [10], 'rfc__n_estimators': [860]
            },
            'abc': {
                'abc__n_estimators': [60]
            },
            'gbc' : {
                'gbc__max_features': [None], 'gbc__min_samples_split': [7], 'gbc__n_estimators': [100]
            },
            'mlp' : {
                'mlp__activation': ['logistic'], 'mlp__learning_rate': ['adaptive'], 'mlp__max_iter': [1000], 'mlp__solver': ['adam']
            },
            'etc' : {
                'etc__max_features': [None], 'etc__min_samples_split': [9], 'etc__splitter': ['random']
            },
            'svc' : {
                'svc__C': [500], 'svc__gamma': [0.0001], 'svc__kernel': ['rbf']
            }
        }

        estimators_combinations = list()
        for n in range(2, len(estimators) + 1):
            estimators_combinations += list(combinations(estimators, n))
        # for combination in estimators_combinations:
        #     logger.info(f'combination : {combination}')

        results = dict()
        for combination in estimators_combinations:
            params = dict()
            path_to_ensemble_combination = f'{path_csv_result}_ensemble'
            for model_tuple in combination:
                params.update(best_params_per_model[model_tuple[0]])
                path_to_ensemble_combination += f'_{model_tuple[0]}'
            
            cv_ensemble = self.cross_validate_ensemble(combination, params, x, y, path_to_ensemble_combination)
            
            combination_key = path_to_ensemble_combination.split("ensemble_")[1]

            logger.info(f'Modelo {combination_key} obteve score {cv_ensemble.best_score_}')

            results[combination_key] = cv_ensemble.best_score_

            del params            
        
        logger.info(f'Fim do gridsearch ensemble. Resultados:')
        for key, val in results.items():
            logger.info(f'{key} : {val}')
        
        return (max(results, key=results.get), results[max(results, key=results.get)])
        

    def grid_search(self, x, y, path_csv_result=None, model_param="SVC"):
        """Define o gridsearch para ser utilizado para valorar os melhores parâmetros para o modelo passado como parâmetro

            Utilizando o parâmetro path_csv_result o resultado completo do GridSearchCV será salvo no path fornecido
        """

        logger.info(f"Iniciando GridSearchCV para o modelo {model_param}")

        time_init = time()

        cross_valid = StratifiedKFold(n_splits=5)

        scoring = { 'auc_score': 'roc_auc',
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


        if model_param == "RF":
            logger.info(f'Usando modelo Random Forest')
            model = RandomForestClassifier()
            params = { 
                'n_estimators': [i for i in range(20, 1001, 20)],
                'max_features': ['sqrt', None],
                'min_samples_split' : [i for i in range(2, 11, 1)]
            }
        elif model_param == "AB":
            logger.info(f'Usando modelo AdaBoost')
            model = AdaBoostClassifier()
            params = {
                'n_estimators' : [i for i in range(20, 1001, 20)]
            }
        elif model_param == "GB":
            logger.info(f'Usando modelo GradientBoost')
            model = GradientBoostingClassifier()
            params = {
                'n_estimators' : [i for i in range(20, 1001, 20)],
                'max_features': ['sqrt', None],
                'min_samples_split' : [i for i in range(2, 11, 1)]
            }
        elif model_param == "MLP":
            logger.info(f'Usando modelo Multilayer Perceptron')
            model = MLPClassifier()
            params = {
                'max_iter' : [500],
                'activation' : ['logistic', 'tanh', 'relu'],
                'solver' : ['lbfgs', 'adam'],
                'learning_rate': ['constant','adaptive']
            }
        elif model_param == "ERT":
            logger.info(f'Usando modelo Extremely Randomized Tree')
            model = ExtraTreeClassifier()
            params = { 
                'max_features': ['sqrt', None],
                'min_samples_split' : [i for i in range(2, 11, 1)],
                'splitter' : ['random', 'best']
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
                                   refit='auc_score',
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
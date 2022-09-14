from genericpath import exists
from pydoc import cli
from numpy import ndarray
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics._scorer import make_scorer
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from utils.setup_logger import logger
from itertools import combinations
from time import time
import json
import os.path  
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import pprint
from scipy import stats
#from IPython.display import display
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt 

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
    
    def plot_grid_search(self, clf):
        """
        Plot as many graphs as parameters are in the grid search results.

        Each graph has the values of each parameter in the X axis and the Score in the Y axis.

        Parameters
        ----------
        clf: estimator object result of a GridSearchCV
            This object contains all the information of the cross validated results for all the parameters combinations.
        """
        # Convert the cross validated results in a DataFrame ordered by `rank_test_score` and `mean_fit_time`.
        # As it is frequent to have more than one combination with the same max score,
        # the one with the least mean fit time SHALL appear first.
        cv_results = pd.DataFrame(clf.cv_results_).sort_values(by=['rank_test_mcc', 'mean_fit_time'])

        # Get parameters
        parameters=cv_results['params'][0].keys()

        # Calculate the number of rows and columns necessary
        rows = -(-len(parameters) // 2)
        columns = min(len(parameters), 2)
        # Create the subplot
        fig = make_subplots(rows=rows, cols=columns)
        # Initialize row and column indexes
        row = 1
        column = 1

        # For each of the parameters
        for parameter in parameters:

            # As all the graphs have the same traces, and by default all traces are shown in the legend,
            # the description appears multiple times. Then, only show legend of the first graph.
            if row == 1 and column == 1:
                show_legend = True
            else:
                show_legend = False

            # Mean test score
            mean_test_score = cv_results[cv_results[f'rank_test_mcc'] != None]
            logger.info(mean_test_score.head(10))
            fig.add_trace(go.Scatter(
                name='Mean test score',
                x=mean_test_score['param_' + parameter],
                y=mean_test_score[f'mean_test_mcc'],
                mode='markers',
                marker=dict(size=mean_test_score['mean_fit_time'],
                            color='SteelBlue',
                            sizeref=2. * cv_results['mean_fit_time'].max() / (40. ** 2),
                            sizemin=4,
                            sizemode='area'),
                text=mean_test_score['params'].apply(
                    lambda x: pprint.pformat(x, width=-1).replace('{', '').replace('}', '').replace('\n', '<br />')),
                showlegend=show_legend),
                row=row,
                col=column)

            # Best estimators
            rank_1 = cv_results[cv_results[f'rank_test_mcc'] == 1]
            fig.add_trace(go.Scatter(
                name='Best estimators',
                x=rank_1['param_' + parameter],
                y=rank_1[f'mean_test_mcc'],
                mode='markers',
                marker=dict(size=rank_1['mean_fit_time'],
                            color='Crimson',
                            sizeref=2. * cv_results['mean_fit_time'].max() / (40. ** 2),
                            sizemin=4,
                            sizemode='area'),
                text=rank_1['params'].apply(str),
                showlegend=show_legend),
                row=row,
                col=column)

            fig.update_xaxes(title_text=parameter, row=row, col=column)
            fig.update_yaxes(title_text='Score', row=row, col=column)

            # Check the linearity of the series
            # Only for numeric series
            if pd.to_numeric(cv_results['param_' + parameter], errors='coerce').notnull().all():
                x_values = cv_results['param_' + parameter].sort_values().unique().tolist()
                r = stats.linregress(x_values, range(0, len(x_values))).rvalue
                # If not so linear, then represent the data as logarithmic
                if r < 0.86:
                    fig.update_xaxes(type='log', row=row, col=column)

            # Increment the row and column indexes
            column += 1
            if column > columns:
                column = 1
                row += 1

                # Show first the best estimators
        fig.update_layout(legend=dict(traceorder='reversed'),
                        width=columns * 360 + 100,
                        height=rows * 360,
                        title='Best score: {:.6f} with {}'.format(cv_results['mean_test_mcc'].iloc[0],
                                                                    str(cv_results['params'].iloc[0]).replace('{',
                                                                                                            '').replace(
                                                                        '}', '')),
                        hovermode='closest',
                        template='none')
        #fig.show()
        with open("p_graph.html", "w") as f:
            f.write(fig.to_html())


    def table_grid_search(self, clf, all_columns=False, all_ranks=False, save=False):
        """Show tables with the grid search results.

        Parameters
        ----------
        clf: estimator object result of a GridSearchCV
            This object contains all the information of the cross validated results for all the parameters combinations.

        all_columns: boolean, default: False
            If true all columns are returned. If false, the following columns are dropped:

            - params. As each parameter has a column with the value.
            - std_*. Standard deviations.
            - split*. Split scores.

        all_ranks: boolean, default: False
            If true all ranks are returned. If false, only the rows with rank equal to 1 are returned.

        save: boolean, default: True
            If true, results are saved to a CSV file.
        """
        # Convert the cross validated results in a DataFrame ordered by `rank_test_score` and `mean_fit_time`.
        # As it is frequent to have more than one combination with the same max score,
        # the one with the least mean fit time SHALL appear first.
        cv_results = pd.DataFrame(clf.cv_results_).sort_values(by=['rank_test_mcc', 'mean_fit_time'])

        # Reorder
        columns = cv_results.columns.tolist()
        # rank_test_score first, mean_test_score second and std_test_score third
        columns = columns[-1:] + columns[-3:-1] + columns[:-3]
        cv_results = cv_results[columns]

        if save:
            cv_results.to_csv('--'.join(cv_results['params'][0].keys()) + '.csv', index=True, index_label='Id')

        # Unless all_columns are True, drop not wanted columns: params, std_* split*
        if not all_columns:
            cv_results.drop('params', axis='columns', inplace=True)
            cv_results.drop(list(cv_results.filter(regex='^std_.*')), axis='columns', inplace=True)
            cv_results.drop(list(cv_results.filter(regex='^split.*')), axis='columns', inplace=True)

        # Unless all_ranks are True, filter out those rows which have rank equal to one
        # if not all_ranks:
        #     cv_results = cv_results[cv_results['rank_test_mcc'] == 1]
        #     cv_results.drop('rank_test_mcc', axis = 'columns', inplace = True)        
        #     cv_results = cv_results.style.hide_index()

        logger.info(cv_results.head())


    def plot_search_results(self, grid, plot_name):
        ## Results from grid search
        results = grid.cv_results_
        means_test = results['mean_test_mcc']
        stds_test = results['std_test_mcc']

        ## Getting indexes of values per hyper-parameter
        masks=[]
        masks_names= list(grid.best_params_.keys())
        for p_k, p_v in grid.best_params_.items():
            masks.append(list(results['param_'+p_k].data==p_v))

        params=grid.param_grid

        ## Ploting results
        fig, ax = plt.subplots(1,len(params),sharex='none', sharey='all',figsize=(20,5))
        fig.suptitle('MCC por parâmetro')
        fig.text(0.04, 0.5, 'MCC', va='center', rotation='vertical')
        pram_preformace_in_best = {}
        for i, p in enumerate(masks_names):
            m = np.stack(masks[:i] + masks[i+1:])
            pram_preformace_in_best
            best_parms_mask = m.all(axis=0)
            best_index = np.where(best_parms_mask)[0]
            x = np.array(params[p])
            if p == "max_depth":
                x[np.where(x == None)] = float('NaN')
            else:
                x[np.where(x == None)] = "None"

            y_1 = np.array(means_test[best_index])
            e_1 = np.array(stds_test[best_index])
            ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test')
            ax[i].set_xlabel(p.upper())

        #plt.legend()
        plt.savefig(f'{plot_name}.png')

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
            logger.info('Ensemble testará GradientBoostingClassifier')
            estimators.append(('xgb', XGBClassifier()))
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

        time_init = time()

        # if testAllCombinations:
        #     estimators_combinations = list()
        #     logger.info(f'Gerando lista de combinações de classificadores')
        #     for n in range(2, len(estimators) + 1):
        #         estimators_combinations += list(combinations(estimators, n))
            
        #     if len(estimators_combinations) < 1:
        #         logger.warn(f'Cancelando execução pois nenhuma combinação foi gerada.')
        #         exit(-1)

        #     for combination in estimators_combinations:
        #         logger.info(f'combination : {combination}')
        # else:
        #     estimators_combinations = [estimators]
        estimators_combinations = [
            [('abc', AdaBoostClassifier()), ('etc', ExtraTreesClassifier())],
            [('gbc', GradientBoostingClassifier()), ('etc', ExtraTreesClassifier())],
            [('abc', AdaBoostClassifier()), ('gbc', GradientBoostingClassifier()), ('etc', ExtraTreesClassifier())]
        ]

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
                'n_estimators' : [5, 10, 25, 50, 100, 200, 300, 400, 500, 750, 1000],
                'algorithm' : ['SAMME', 'SAMME.R'],
                'learning_rate' : [1, 0.5, 0.25, 0.1, 0.05, 0.01]
            }
        elif model_param == "gbc":
            logger.info(f'Usando modelo GradientBoost')
            model = GradientBoostingClassifier()
            params = {
                'n_estimators' : [5, 10, 25, 50, 100, 200, 300, 400, 500, 750, 1000],
                'max_features': ['sqrt', None],
                'learning_rate' : [1, 0.5, 0.25, 0.1, 0.05, 0.01],
                #'subsample' : [0.5, 0.6, 0.7, 0.8, 0.9],
                'max_depth': [3, 5, 8, 16]
            }
        elif model_param == "xgb":
            logger.info(f'Usando modelo XGBClassifier')
            model = XGBClassifier(verbosity=2, tree_method="hist")
            # params = {
            #     'n_estimators' : [5, 10, 25, 50, 100, 200, 300, 400, 500, 750, 1000],
            #     'learning_rate' : [1, 0.5, 0.25, 0.1, 0.05, 0.01],
            #     'max_depth': [3, 5, 8, 16],
            #     'min_child_weight' : [1, 3, 5]
            #
            params = {
                'n_estimators' : [500],
                'learning_rate' : [0.01],
                'max_depth': [3],
                'min_child_weight' : [5],
                'subsample' : [0.6, 0.7, 0.8, 0.9],
                'colsample_bytree' : [0.6, 0.7, 0.8, 0.9],
                'gamma' : [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1],
                'reg_alpha': [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1]
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
                'n_estimators' : [5, 10, 25, 50, 100, 200, 300, 400, 500, 750, 1000],
                'max_features': ['sqrt', 'log2', None],
                'min_samples_split' : [2, 8, 16, 32, 64, 128],
                'max_depth': [2, 8, 16, None]
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
            elif model == 'xgb':
                logger.info('Ensemble testará XGBClassifier')
                estimators.append(('xgb', XGBClassifier()))    
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

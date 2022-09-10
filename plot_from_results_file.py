import sys
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import pprint
from scipy import stats
import ast
from itertools import combinations

def plot_ensemble(f_name):
    print(f'Cheguei no plot_ensemble com parametro f_name = {f_name}')

    estimators = ['abc', 'gbc', 'etc'] if 'gbc' in f_name else ['abc', 'xgb', 'etc']
    columns_needed = ['mean_fit_time', 'mean_score_time', 'mean_test_auc_score', 'mean_test_accuracy', 'mean_test_scores_p_1', 'mean_test_scores_r_1', 'mean_test_scores_f_1_1', 'mean_test_scores_p_0', 'mean_test_scores_r_0', 'mean_test_scores_f_1_0', 'mean_test_mcc', 'mean_test_precision_micro', 'mean_test_precision_macro', 'mean_test_recall_macro', 'mean_test_recall_micro', 'mean_test_f1_macro', 'mean_test_f1_micro']
    print(f'len(columns_needed) = {len(columns_needed)}')
    scoring_metrics = ['auc_score', 'accuracy', 'scores_p_1', 'scores_r_1', 'scores_f_1_1', 'scores_p_0', 'scores_r_0', 'scores_f_1_0', 'mcc', 'precision_micro', 'precision_macro', 'recall_macro', 'recall_micro', 'f1_macro', 'f1_micro']
    print(f'len(scoring_metrics) = {len(scoring_metrics)}')

    estimators_combinations = []
    for n in range(2, len(estimators) + 1):
                estimators_combinations += list(combinations(estimators, n))

    estimators_combinations = ["_".join(c) for c in [list(combination) for combination in estimators_combinations]]
    print(estimators_combinations)

    df = pd.read_csv(f'{f_name}{estimators_combinations[0]}')[columns_needed]
    

    for combination in estimators_combinations[1:]:
        aux = pd.read_csv(f'{f_name}{combination}')[columns_needed]
        df = pd.concat([df, aux], axis=0, ignore_index=True)

    df["ensemble_name"] = estimators_combinations   

    for metric in scoring_metrics:
        df[f'rank_test_{metric}'] = df[f'mean_test_{metric}'].rank(ascending=False)

    print(df.head(10))

    df = df.sort_values(by=['rank_test_mcc', 'mean_fit_time'])

    # Calculate the number of rows and columns necessary
    rows = -(-len(estimators_combinations) // 2)
    columns = min(len(estimators_combinations), 2)
    # Create the subplot
    fig = make_subplots(rows=rows, cols=columns)
    # Initialize row and column indexes
    row = 1
    column = 1

    # Mean test score
    mean_test_score = df[df[f'rank_test_mcc'] != None]
    fig.add_trace(go.Scatter(
        name='Mean test score',
        x=mean_test_score["ensemble_name"],
        y=mean_test_score[f'mean_test_mcc'],
        mode='markers',
        marker=dict(size=mean_test_score['mean_fit_time'],
                    color='SteelBlue',
                    sizeref=2. * df['mean_fit_time'].max() / (40. ** 2),
                    sizemin=4,
                    sizemode='area'),
        text=mean_test_score['ensemble_name'].apply(
            lambda x: pprint.pformat(x, width=-1).replace('{', '').replace('}', '').replace('\n', '<br />')),
        showlegend=False),
        row=row,
        col=column)

    # Best estimators
    rank_1 = df[df[f'rank_test_mcc'] == 1]
    fig.add_trace(go.Scatter(
        name='Best estimators',
        x=rank_1["ensemble_name"],
        y=rank_1[f'mean_test_mcc'],
        mode='markers',
        marker=dict(size=rank_1['mean_fit_time'],
                    color='Crimson',
                    sizeref=2. * df['mean_fit_time'].max() / (40. ** 2),
                    sizemin=4,
                    sizemode='area'),
        text=rank_1['ensemble_name'].apply(str),
        showlegend=False),
        row=row,
        col=column)

    fig.update_xaxes(title_text="Ensemble", row=row, col=column)
    fig.update_yaxes(title_text='Score', row=row, col=column)

    fig.update_layout(legend=dict(traceorder='reversed'),
                width=columns * 360 + 100,
                height=rows * 360,
                title='Best score: {:.6f} with {}'.format(df['mean_test_mcc'].iloc[0],
                                                            str(df['ensemble_name'].iloc[0]).replace('{',
                                                                                                    '').replace(
                                                                '}', '')),
                hovermode='closest',
                template='none')
    
    #fig.show()
    with open(f'{f_name}_graph.html', "w") as f:
        f.write(fig.to_html())

def plot_grid_search(f_name):
    print(f'Cheguei no plot_grid_search com parametro f_name = {f_name}')
    cv_results = pd.read_csv(f_name).sort_values(by=['rank_test_auc_score', 'mean_fit_time'])
    
    # Convert the cross validated results in a DataFrame ordered by `rank_test_score` and `mean_fit_time`.
    # As it is frequent to have more than one combination with the same max score,
    # the one with the least mean fit time SHALL appear first.
    #cv_results = pd.DataFrame(clf.cv_results_).sort_values(by=['rank_test_auc_score', 'mean_fit_time'])

    # Get parameters
    parameters=ast.literal_eval(cv_results['params'][0]).keys()

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
        mean_test_score = cv_results[cv_results[f'rank_test_auc_score'] != None]
        fig.add_trace(go.Scatter(
            name='Mean test score',
            x=mean_test_score['param_' + parameter],
            y=mean_test_score[f'mean_test_auc_score'],
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
        rank_1 = cv_results[cv_results[f'rank_test_auc_score'] == 1]
        fig.add_trace(go.Scatter(
            name='Best estimators',
            x=rank_1['param_' + parameter],
            y=rank_1[f'mean_test_auc_score'],
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
                    title='Best score: {:.6f} with {}'.format(cv_results['mean_test_auc_score'].iloc[0],
                                                                str(cv_results['params'].iloc[0]).replace('{',
                                                                                                        '').replace(
                                                                    '}', '')),
                    hovermode='closest',
                    template='none')
    #fig.show()
    with open(f'{f_name}_graph.html', "w") as f:
        f.write(fig.to_html())

if __name__=='__main__':
    #plot_grid_search(sys.argv[1])
    plot_ensemble(sys.argv[1])
# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.


#
#
# v3.14 - add tissue selection checkboxes
# v3.14.rapids - implement cudf for major speed improvement
# 
#

import dash
from dash import dash_table as dt
from dash.dash_table.Format import Format, Scheme, Trim
from dash import html
from dash import dcc
import plotly.express as px
#import pandas as pd
import cudf as pd
import numpy as np 
import datetime
from dash.dependencies import Input, Output, State
from dash import no_update
#from html import unescape
import json

DEV_INSTANCE=False  # is this the dev instance or the production one?
#
# SET TO FALSE IN PRODUCTION INSTANCE
#

file_path = '/var/www/dash/Data/pickles4_avana.txt'
df = pd.read_csv(file_path,
    dtype={ 
       'Gene':str,
       'Avana_Z':'float32',
       'Avana_BF':'float32',
       'Avana_Chronos':'float32',
       'Expr':'float32',
       'stripped_cell_line_name': str,
       'primary_disease': str,
       'lineage': str,
       'GOF': bool,
       'LOF': bool
   },
   sep='\t').dropna( subset=['Gene','Avana_Z','Avana_BF','Avana_Chronos'], how='any')
   
#df = pd.read_table('/var/www/dash/Data/master_table_Avana_Score_TKOv3_BF_Zscore_Expr_LOF_GOF_18959genes_1162cells_lineage_disease.txt', 


unique_lineages = sorted( list( set( df['primary_disease'].values_host ) ) )
unique_genes = sorted( list( set( df['Gene'].values_host ) ) )

#gene_list_dropdown = [ {'label': i, 'value': i } for i in list( df.Gene.unique().values_host )]
gene_list_dropdown = [ {'label': i, 'value': i } for i in unique_genes ]

gene1_default = 'BRAF'
gene2_default = 'RAF1'


#
# initialize matrices for coessentiality calcs
#
my_columns = ['Avana_BF','Avana_Z','Avana_Chronos']

#
# initialize matrices for coexpression calcs
#
#expression_matrices = []
#expression_matrix = df.drop(df.index.values[np.isnan(df['Expr'].values)], axis=0).pivot(
#    index='Gene', columns='stripped_cell_line_name', values='Expr')


my_column_default = 'Avana_BF'


logfile = open('./dash_usage_log_rapids.txt', 'a', buffering=1)


app = dash.Dash(__name__)


if (DEV_INSTANCE):
    app.title = 'Pickles3_dev'
else:
    app.title = 'Pickles3'

def cholesky_covariance_warp(df):
    """
    Perform cholesky decomposition and "covariance warping" on a genes (rows)
     by samples (columns) dataframe.
    returns a dataframe with the same index and columns.
    """
    # first, drop missing data:
    df.dropna(axis=0, how='any', inplace=True)
    # 
    # then do the inv, etc.
    #
    cholsigmainv = np.linalg.cholesky(np.linalg.pinv(np.cov(df.T)))
    warped_screens = df.values_host @ cholsigmainv
    cholesky_df = pd.DataFrame( warped_screens , index=df.index.values, columns=df.columns.values)
    return cholesky_df

def corr_one_vs_all(gene, bf_mat):
    """Correlate one gene essentiality scores vs. all other genes, cheap and fast.
    Parameters
    ----------
    gene - the gene name to extract
    
    bf_mat: dataframe, the N x M matrix of gene essentiality scores
        N = num gdenes
        M = num samples
    
    Returns
    -------
    pandas.dataframe
      N-1 x 1 frame in which each element is a Pearson correlation coefficient. 
      Index = bf_mat index labels, excluding 'gene'
      Columns = 'gene' parameter
    """
    normed_matrix = bf_mat
    #normed_matrix = cholesky_covariance_warp(bf_mat2)
    normed_matrix.dropna(axis=0, how='any', inplace=True)

    gene_vector   = normed_matrix.loc[[gene]]
    other_vectors = normed_matrix.drop(gene, axis=0)
    n = gene_vector.shape[1]
    mu_x = gene_vector.mean(1)
    mu_y = other_vectors.mean(1)
    s_x = gene_vector.std(1, ddof=n - 1)
    s_y = other_vectors.std(1, ddof=n - 1)
    cov = np.dot(gene_vector.values_host, other_vectors.T.values_host) - n * (mu_x.values_host * mu_y.values_host)
    corrmatrix = cov / (s_x.values_host * s_y.values_host)
    outframe = pd.DataFrame( index=other_vectors.index.values_host, columns=[gene], data=corrmatrix.T )
    return outframe

def corr_one_vs_Expr_all(gene, bf_mat, expr_mat):
    """Correlate one gene essentiality scores vs. all other genes expression TPM.
    Parameters
    ----------
    gene - the gene name to extract   
    bf_mat: dataframe, the N x M matrix of gene essentiality scores
        N = num genes
        M = num samples
    expr_mat: dataframe, the N x M matrix of gene expression TPM
        N = num genes
        M = num samples

    Returns
    -------
    pandas.dataframe
      N-1 x 1 frame in which each element is a Pearson correlation coefficient. 
      Index = expr_mat index labels, excluding 'gene'
      Columns = 'gene' parameter
    """

    gene_essentiality_vector = bf_mat.loc[[gene]]
    #other_vectors = expr_mat.drop(gene, axis=0) # why drop current gene???
    expr_mat.dropna(axis=0, how='any', inplace=True)    # missing values will fuck us up

    # Match the columns of bf_mat & expr_mat
    matched_columns = np.intersect1d(
        gene_essentiality_vector.columns, expr_mat.columns)
    gene_essentiality_vector = gene_essentiality_vector[matched_columns]
    expr_mat = expr_mat[matched_columns]
    n = gene_essentiality_vector.shape[1]

    mu_x = gene_essentiality_vector.mean(1)
    mu_y = expr_mat.mean(1)
    s_x = gene_essentiality_vector.std(1, ddof=n - 1)
    s_y = expr_mat.std(1, ddof=n - 1)
    cov = np.dot(gene_essentiality_vector.values_host, expr_mat.T.values_host) - \
        n * (mu_x.values_host * mu_y.values_host)
    corrmatrix = cov / (s_x.values_host * s_y.values_host)
    outframe = pd.DataFrame(index=expr_mat.index.values_host, columns=[gene], data=corrmatrix.T)
    return outframe


#####
# MAIN PAGE LAYOUT
#####

app.layout = html.Div(className='container', children=[ 

    #################
    # side info bar #
    #################
    html.Div(  className='NavBar', children=[ 
        html.H2('PICKLES v3'),
        html.Label('Which dataset/score?'),
        dcc.RadioItems(
            options=[
                {'label': 'Avana/Chronos', 'value': 'Avana_Chronos'},
                {'label': 'Avana/BF', 'value': 'Avana_BF'},
                {'label': 'Avana/Z score', 'value': 'Avana_Z'},
                #{'label': 'Score/BF', 'value': 'Score_BF'},
                #{'label': 'Score/Z score', 'value': 'Score_Z'},
                #{'label': 'TKOv3/BF', 'value': 'TKOv3_BF'},
                #{'label': 'TKOv3/Z score', 'value': 'TKOv3_Z'},
                  
            ],
            value='Avana_BF',
            id='dataset',
            persistence=True,
            persistence_type='session',
            labelStyle={'display': 'block'},
        ),
        html.Br(),
        #html.Label('Which scoring scheme?'),
        #dcc.RadioItems(
        #    options=[
        #        {'label': 'Bayes Factor', 'value': 'BF'},
        #        {'label': 'Z-score', 'value': 'Z'},
        #    ],
        #    value='BF',
        #    id='algo',
        #    persistence=True,
        #    persistence_type='local',
        #    labelStyle={'display': 'block'},

        #),
        html.Br(),

        html.Label('Query gene:'),
        #dcc.Input(value='BRAF', type='text', id='gene1'),
        dcc.Dropdown( id='gene1',
            options=gene_list_dropdown,
            value=gene1_default,
            persistence=True,
            persistence_type='local',
            style={'width':'125px','textAlign':'left'}),
        html.Br(),
        html.Br(),

        html.Div( id='InfoAreaContextText', children =[
                html.Div( id='info_area_context_text_content' ),
                dcc.Dropdown( id='gene2',
                    options=gene_list_dropdown,
                    value=gene2_default,
                    persistence=True,
                    persistence_type='local',
                    style={'width':'125px','textAlign':'left'}),
                html.Br(),
                ], style={'width':'100%','textAlign':'left', 'display': 'none'}),
                #], style={'width':'100%','textAlign':'left'}),
            #]),
        html.Br(),
        html.Div( id='TissueSelection', children=[
            html.Label('Select tissues:'),
            dcc.Checklist(
                id='selected_lineages',
                options=[{'label': lineage, 'value': lineage} for lineage in unique_lineages],
                value=unique_lineages,  # All lineages selected by default
                persistence=True,
                persistence_type='session',
                style={'width':'125px','textAlign':'left', 'font-size': '10px'}),
            html.Br(),
            ]),
   
        html.Div( id='link_to_Hartlab', children=[
            dcc.Link( 'Hart Lab', href='https://www.hart-lab.org')
            ], style={'width':'100%','textAlign':'center','position':'absolute','bottom':0, 'font-size': '18px'}),
            # all this style 'position''bottom' etc. puts the link at the bottom of the left bar div
            # requires 'position''relative' in parent div; see below

    ], style={'textAlign':'left', 'position':'relative'}),


    ###############
    # top tab bar #
    ###############
    html.Div( className='TabBar', children=[
        dcc.Tabs(id='graph_type', value='summary', children=[
            dcc.Tab(label='Summary', value='summary'),
            dcc.Tab(label='Cancer Types', value='tissue'),
            dcc.Tab(label='Co-essentiality', value='coess'),
            dcc.Tab(label='Expression', value='expr'),
            dcc.Tab(label='Mutation', value='muts'),
            dcc.Tab(label='About', value='about'),
            #dcc.Tab(label='Copy Number', value='cna'),
            ]),
        ]),

    ################
    # display area #
    ################
    html.Div( className='PlotArea', children=[
        #
        html.Div( id='main_figure_div', children =[
            dcc.Graph(id='main_figure')
            ], style={'width':'100%','textAlign':'center'}),
        html.Div( id='DisplayAreaText', children =[
            html.Div( id='display_area_text' ),
            ], style={'width':'100%','textAlign':'left', 'display': 'none'}),
        ]),
])

###
# end of layout. define callbacks.
###

@app.callback(
    Output('main_figure', 'figure'),
    Output('main_figure_div','style'),
    Output('info_area_context_text_content', component_property='children'),
    Output('InfoAreaContextText', 'style'),
    Output('display_area_text', component_property='children'),
    Output('DisplayAreaText', 'style'),
    Input('dataset','value'),
    #Input('algo', 'value'),
    Input('gene1', 'value'),
    Input('gene2', 'value'),
    Input('selected_lineages', 'value'),
    Input('graph_type', 'value'))

#def update_figure(dataset, algo, gene1, gene2, graph_type):
def update_figure(dataset, gene1, gene2, selected_lineages, graph_type):
    style0 = {'width':'100%','textAlign':'center'}
    style1 = {'width':'100%','textAlign':'left', 'display': 'none'}
    style2 = {'width':'100%','textAlign':'left', 'display': 'none'}
    #gene1 = gene1.upper()
    #gene2 = gene2.upper()
    if (not gene2):
        gene2 = gene1
    #my_column = dataset + '_' + algo
    my_column = dataset
    algo = dataset.split('_')[1]
    info_area_context_text_content = ''
    display_area_context_text_content = ''
    plot_width = 800
    plot_height = 640
    nowstring = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logstring = nowstring + '\t' + my_column + '\t' + graph_type + '\t' + gene1 + '\t' + gene2 + '\n'
    logfile.write(logstring)

    if ('_BF' in my_column):
        ascend = False
    else:  
        ascend = True

    showlegend = True




    if (graph_type == 'summary'):
        #
        ########################################
        # "SUMMARY" tab 
        #   - hide comparison gene
        #   - waterfall plot: query gene essentiality, screens ranked by score
        #
        #
        ########################################
        #        
        d1 = df[ (df.Gene==gene1) & df['primary_disease'].isin(selected_lineages) ]\
                .sort_values(my_column, ascending=ascend)\
                .dropna(axis=0, subset=[my_column])
        #cells_not_assayed = np.where( np.isnan( d1[my_column]) )[0]
        #d1.dropna(axis=0, how='all', inplace=True, subset=[my_column])      # dro   
        d1['idx'] = range(1,d1.shape[0]+1)
        fig = px.scatter(d1, x='idx', y=my_column,
                     hover_name='stripped_cell_line_name',
                     labels = {
                         'idx': 'Cell line rank',
                         my_column: gene1 + ', ' + algo,
                         'primary_disease' : 'Primary Disease'
                     },
                     category_orders = {
                        'primary_disease' : sorted( list( d1.primary_disease.values_host ) )
                     },
                     color='primary_disease',
                     width=plot_width, height=plot_height)



    elif (graph_type == 'tissue'):
        #
        ########################################
        # "CANCER TYPES" tab 
        #   - hide comparison gene
        #   - stripplot of query gene essentialty by primary disease
        #   - calculate 
        ########################################
        #        
        d1 = df[ (df.Gene==gene1) & (df['primary_disease'].isin(selected_lineages)) ].dropna(axis=0, subset=[my_column])
        #cells_not_assayed = np.where( np.isnan( d1[my_column]) )[0]
        #d1.drop( d1.index.values[ cells_not_assayed ], axis=0, inplace=True)    
        sorted_list = d1[ [my_column, 'primary_disease'] ].groupby('primary_disease').median().sort_values(my_column, ascending=ascend )
        disease_rank = pd.DataFrame( index=sorted_list.index.values_host, columns=['Rank'], data=range(1,len(sorted_list)+1) )
        fig = px.strip(d1, x='primary_disease', y=my_column,
                     hover_name='stripped_cell_line_name',
                     labels = {
                         'median_sort_rank': 'disease type',
                         my_column: gene1 + ', ' + algo + ' (' + dataset + ')',
                         'primary_disease' : 'Primary Disease'
                     },
                     category_orders = {
                        'primary_disease' : list( disease_rank.index.values_host)
                     },
                     color='primary_disease',
                     width=plot_width, height=plot_height) 
                     #size_max=55)
        showlegend=False

    elif (graph_type == 'coess'):
        #
        ########################################
        # "COESSENTIAL" tab 
        #   - show comparison gene
        #   - scatterplot of query gene essentialty vs comparison gene essentiality
        #   - calculate 
        ########################################
        #
        d1 = df[ (df.Gene==gene1) & (df['primary_disease'].isin(selected_lineages) )][[my_column,'stripped_cell_line_name','primary_disease']]
        if (gene1 != gene2):
            d2 = df[ (df.Gene==gene2) & (df['primary_disease'].isin(selected_lineages) ) ][[my_column,'stripped_cell_line_name']]
            d = d1.merge(d2, on='stripped_cell_line_name', suffixes=("_" + gene1, "_" + gene2))
        else:
            d = d1.copy()
            d.columns = [my_column + "_" + gene1, 'stripped_cell_line_name','primary_disease']
            
        fig = px.scatter(d, x=my_column + '_' + gene1, 
                     y=my_column + '_' + gene2,
                     hover_name='stripped_cell_line_name',
                     trendline='ols',
                     trendline_scope='overall',
                     trendline_color_override='lightgray',
                     labels = {
                         my_column + '_' + gene1:  gene1 + ', ' + algo + ' (' + dataset + ')',
                         my_column + '_' + gene2:  gene2 + ', ' + algo + ' (' + dataset + ')',
                         'primary_disease' : 'Primary Disease'
                     },
                     category_orders = {
                        'primary_disease' : sorted( list( d.primary_disease.values_host ) )
                     },
                     color='primary_disease',
                     width=plot_width, height=plot_height)  
                     #size_max=55)
        style1 = {'width':'100%','textAlign':'left'}
        info_area_context_text_content = 'Comparison gene:'
        #
        ####
        # calculate and display top correlates with query gene.
        # do we need to do this every time a seconardy gene is selected?
        # not sure how to make this more efficient
        ####
        #
        # calculate all corrs
        #
        ess_matrix = df.drop( df.index.values[ np.isnan( df[my_column].values_host) ], axis=0 )
        ess_matrix = ess_matrix[ ess_matrix['primary_disease'].isin( selected_lineages) ].pivot( 
            index='Gene', columns='stripped_cell_line_name', values=my_column )

        my_correlates = corr_one_vs_all( gene1, ess_matrix )
        my_correlates.columns = ['PCC']
        my_correlates.index.name = gene1 
        top_corrs = my_correlates.nlargest(100, 'PCC')
        bot_corrs = my_correlates.nsmallest(100, 'PCC')
        # 
        display_area_context_text_content = html.Div(children=[

            html.Div(id='top_correlates', children = [
                    html.H3('Top positive correlates'),
                    dt.DataTable(
                        data=top_corrs.reset_index().to_dict('records'),
                        #columns=[{'name': i, 'id': i} for i in top_corrs.reset_index().columns],
                        columns=[
                            dict(id=gene1, name=gene1),
                            dict(id='PCC', name='PCC',type='numeric',format=Format(precision=3, scheme=Scheme.fixed))
                            ],
                        fixed_rows={'headers': True},
                        page_size=10,
                        style_header={'fontSize':'14px', 'font-family':'sans-serif','fontWeight': 'bold','backgroundColor': '#e1e4eb'},
                        style_cell={'minWidth': 100, 'width': 100, 'maxWidth': 100,'fontSize':'12px', 'font-family':'sans-serif'},
                        #style_table={'height': 400, 'width':220}
                        style_table={'width':220}
                        )
                    ], style={'display':'inline-block', 'padding-right':'5px'}),
                html.Div(id='bot_correlates', children = [
                    html.H3('Top negative correlates'),
                    dt.DataTable(
                        data=bot_corrs.reset_index().to_dict('records'),
                        columns=[
                            dict(id=gene1, name=gene1),
                            dict(id='PCC', name='PCC',type='numeric',format=Format(precision=3, scheme=Scheme.fixed))
                            ],
                        fixed_rows={'headers': True},
                        page_size=10,
                        style_header={'fontSize':'14px', 'font-family':'sans-serif','fontWeight': 'bold','backgroundColor': '#e1e4eb'},
                        style_cell={'minWidth': 100, 'width': 100, 'maxWidth': 100,'fontSize':'12px', 'font-family':'sans-serif'},
                        #style_table={'height': 400, 'width':220}
                        style_table={'width':220}
                        )
                    ], style={'display':'inline-block','padding-left':'5px'}),
                ])
        style2 = {'width':'100%','textAlign':'left', 'height':450}

    elif (graph_type == 'expr'):

        if ('TKOv3' in my_column):
            #
            ########################################
            # TKOv3 does not have expression data
            ########################################
            #
            fig = px.scatter(x=[1], y=[1], width=10, height=10)
            style0 = {'width': '100%', 'textAlign': 'left', 'display': 'none'}
            display_area_context_text_content = html.Div(children=[
                html.P('Expression data not available for TKOv3 screens',
                       style={'font-weight': 'bold'})
            ])
            style2 = {'width': '100%', 'textAlign': 'left', 'height': 450}
            essStyle = {'display':'none'}

        else:
            #
            ########################################
            # "EXPRESSION" tab 
            #   - show comparison gene
            #   - scatterplot of query gene essentialty vs comparison gene expression
            #   - TODO: show correlates?
            ########################################
            #
            d1 = df[ (df.Gene==gene1) & (df['primary_disease'].isin(selected_lineages) ) ][[my_column,'stripped_cell_line_name']]
            d1.columns = [my_column + '_' + gene1, 'stripped_cell_line_name']
            d2 = df[ (df.Gene==gene2) & (df['primary_disease'].isin(selected_lineages) ) ][['Expr','stripped_cell_line_name','primary_disease']]
            d2.columns = ['Expr_' + gene2, 'stripped_cell_line_name','primary_disease']
            d = d1.merge(d2, on='stripped_cell_line_name')
            
            fig = px.scatter(d, x=my_column + '_' + gene1, 
                         y='Expr_' + gene2,
                         hover_name='stripped_cell_line_name',
                         trendline='ols',
                         trendline_scope='overall',
                         trendline_color_override='lightgray',
                         labels = {
                             my_column + '_' + gene1:  gene1 + ', ' + algo + ' (' + dataset + ')',
                             'Expr_' + gene2:  gene2 + ', log(TPM)',
                             'primary_disease' : 'Primary Disease'
                         },
                         category_orders = {
                            'primary_disease' : sorted( d.primary_disease.values_host )
                         },
                         color='primary_disease',
                         width=plot_width, height=plot_height)  
                         #size_max=55)
            style1 = {'width':'100%','textAlign':'left'}
            info_area_context_text_content = 'Comparison gene:'

            #
            ####
            # calculate and display top expression correlates with query gene.
            ####
            #
            # calculate all corrs
            #
            ess_table = df.dropna(axis=0, subset=[my_column, 'Expr'])
            ess_matrix = ess_table[ ess_table['primary_disease'].isin( selected_lineages) ].pivot( 
                index='Gene', columns='stripped_cell_line_name', values=my_column )
            expression_matrix = ess_table[ ess_table['primary_disease'].isin( selected_lineages) ].pivot( 
                index='Gene', columns='stripped_cell_line_name', values='Expr' )
            expression_matrix = expression_matrix.loc[ expression_matrix.max(1) > 2 ]

            my_correlates = corr_one_vs_Expr_all( gene1, ess_matrix, expression_matrix )
            my_correlates.columns = ['PCC']
            my_correlates.index.name = gene1 
            top_corrs = my_correlates.nlargest(100, 'PCC')
            bot_corrs = my_correlates.nsmallest(100, 'PCC')
            # 
            display_area_context_text_content = html.Div(children=[

                html.Div(id='top_correlates', children = [
                    html.H3('Top positive correlates'),
                    dt.DataTable(
                        data=top_corrs.reset_index().to_dict('records'),
                        #columns=[{'name': i, 'id': i} for i in top_corrs.reset_index().columns],
                        columns=[
                            dict(id=gene1, name=gene1),
                            dict(id='PCC', name='PCC',type='numeric',format=Format(precision=3, scheme=Scheme.fixed))
                            ],
                        fixed_rows={'headers': True},
                        page_size=10,
                        style_header={'fontSize':'14px', 'font-family':'sans-serif','fontWeight': 'bold','backgroundColor': '#e1e4eb'},
                        style_cell={'minWidth': 100, 'width': 100, 'maxWidth': 100,'fontSize':'12px', 'font-family':'sans-serif'},
                        #style_table={'height': 400, 'width':220}
                        style_table={'width':220}
                        )
                    ], style={'display':'inline-block', 'padding-right':'5px'}),
                html.Div(id='bot_correlates', children = [
                    html.H3('Top negative correlates'),
                    dt.DataTable(
                        data=bot_corrs.reset_index().to_dict('records'),
                        columns=[
                            dict(id=gene1, name=gene1),
                            dict(id='PCC', name='PCC',type='numeric',format=Format(precision=3, scheme=Scheme.fixed))
                            ],
                        fixed_rows={'headers': True},
                        page_size=10,
                        style_header={'fontSize':'14px', 'font-family':'sans-serif','fontWeight': 'bold','backgroundColor': '#e1e4eb'},
                        style_cell={'minWidth': 100, 'width': 100, 'maxWidth': 100,'fontSize':'12px', 'font-family':'sans-serif'},
                        #style_table={'height': 400, 'width':220}
                        style_table={'width':220}
                        )
                    ], style={'display':'inline-block','padding-left':'5px'}),
                ])
            style2 = {'width':'100%','textAlign':'left', 'height':450}



    elif (graph_type == 'muts'):

        if ('TKOv3' in my_column):
            fig = px.scatter(x=[1], y=[1], width=10, height=10)
            style0 = {'width': '100%', 'textAlign': 'left', 'display': 'none'}
            display_area_context_text_content = html.Div(children=[
                html.P('Mutation data not available for TKOv3 screens',
                       style={'font-weight': 'bold'})
            ])
            style2 = {'width': '100%', 'textAlign': 'left', 'height': 450}
        else:
            #
            ########################################
            # "MUTATIONS" tab 
            #   - show comparison gene
            #   - strip plot by category of comparison gene
            ########################################
            #
            d1 = df[ (df.Gene==gene1) & (df['primary_disease'].isin(selected_lineages) ) ][[my_column,'stripped_cell_line_name','primary_disease']]
            d2 = df[ (df.Gene==gene2) & (df['primary_disease'].isin(selected_lineages) ) ][['LOF','GOF','stripped_cell_line_name']]
            d2['MUT'] = 'WT'
            #d2.loc[ d2[ d2.GOF==True].index.values, 'MUT'] = 'GOF'
            d2.loc[ d2[ d2.GOF==True].index.values_host, 'MUT'] = 'Hotspot'
            d2.loc[ d2[ d2.LOF==True].index.values_host, 'MUT'] = 'LOF'
            d = d1.merge(d2, on='stripped_cell_line_name', how='inner')
            
            fig = px.strip(d, x='MUT',
                         y=my_column,
                         hover_name='stripped_cell_line_name',
                         labels = {
                             my_column:  gene1 + ', ' + algo + ' (' + dataset + ')',
                             'MUT':  gene2 + ' mutation state',
                             'primary_disease' : 'Primary Disease'
                         },
                         category_orders = {
                            'primary_disease' : sorted( d.primary_disease.values_host )
                         },
                         color='primary_disease',
                         width=plot_width, height=plot_height)  
                         #size_max=55)
            style1 = {'width':'100%','textAlign':'left'}
            info_area_context_text_content = 'Comparison gene:'



    elif (graph_type == 'about'):
        #
        ########################################
        # "ABOUT" tab 
        #   - hide main figure
        #   - no change to comparison gene
        #   - show block of 'about' text
        #   - link to download file
        ########################################
        #
        fig = px.scatter(x=[1], y=[1], width=10, height=10)
        style0 = {'width':'100%','textAlign':'left', 'display': 'none'}
        #
        # add display area text
        #
        display_area_context_text_content = html.Div(children=[
            html.H2('About PICKLES v3'),
            html.P('Dataset', style={'font-weight':'bold'}),
            html.Ul(className='about_dataset', children=[
                html.Li('Avana/Broad: DepMap cell lines screened with the Avana library (20Q4 release)'),
                html.Li('Score/Sanger: Project Score cell lines screened with the Yusa library (Behan et al Nature 2019)'),
                html.Li('TKOv3/various: TKOv3 (Hart et al G3 2017) screens from several publications. Expression and mutation data not available.')
                ]),
            html.P('Scoring scheme', style={'font-weight':'bold'}),
            html.Ul(className='about_scoring', children=[
                html.Li('Bayes Factor: BAGEL2 scoring from Kim & Hart, Genome Medicine, 2021. Positive scores = essential.'),
                html.Li('Z-score: Gaussian mixture model Z-scores from Lenoir et al, Nature Communications 2021. Negative scores=essential')
                ]),
            html.P('Query gene', style={'font-weight':'bold'}),
            html.Ul(className='about_querygene', children=[
                html.Li('the gene selected for fitness profiling')
                ]),
            html.P('Comparison gene', style={'font-weight':'bold'}),
            html.Ul(className='about_comparisongene', children=[
                html.Li('the query gene\'s fitness is profiled against the comparison gene\'s fitness, expression, or mutation. Can be the same gene.')
                ]),
            html.P(''),
            html.P(children=[
                'Download master data file used in this application (Caution: 1.65GB!):  ', 
                dcc.Link( 'Master data file', 
                    href='https://neptune.hart-lab.org/Downloads/master_table_Avana_Score_TKOv3_BF_Zscore_Expr_LOF_GOF_19731genes_1015cells_lineage_disease.txt', 
                    style={'font-weight': 'bold'}),
                ]),
            ])
        style2 = {'width':'100%','textAlign':'left'}
        


    else:
        fig = px.scatter(x=[1], y=[1], width=10, height=10)

    fig.update_layout(transition_duration=500, showlegend=showlegend, font_size=10,)


    return fig, style0, info_area_context_text_content, style1, display_area_context_text_content, style2


if __name__ == '__main__':
    if (DEV_INSTANCE):
        app.run_server(debug=True, port=8050)
    else:
        app.run_server(debug=False)


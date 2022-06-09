# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import plotly.express as px
import pandas as pd
import numpy as np 
import datetime
from dash.dependencies import Input, Output
from html import unescape

DEV_INSTANCE=True  # is this the dev instance or the production one?
#
# SET TO FALSE IN PRODUCTION INSTANCE
#


df = pd.read_table('/var/www/dash/Data/master_table_Avana_Score_TKOv3_BF_Zscore_Expr_LOF_GOF_19731genes_1015cells_lineage_disease.txt', 
    dtype={ 
       'Gene':str,
       'Cell_Line':str,
       'stripped_cell_line_name': str,
       'primary_disease': str,
       'lineage': str
       #'GOF': bool,
       #'LOF': bool
   },
   sep='\t',
   index_col=0)


gene_list_dropdown = [ {'label': i, 'value': i } for i in list( df.Gene.unique() )]

gene1_default = 'BRAF'
gene2_default = 'RAF1'


#
# initialize matrices for coessentiality calcs
#
my_columns = ['Avana_BF','Avana_Z','Score_BF','Score_Z','TKOv3_BF','TKOv3_Z']
essentiality_matrices = {}
for col in my_columns:
    essentiality_matrices[col] = df.drop( df.index.values[ np.isnan( df[col].values) ], axis=0 ).pivot( index='Gene', columns='stripped_cell_line_name', values=col)

logfile = open('./dash_usage_log.txt', 'a', buffering=1)

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
    warped_screens = df.values @ cholsigmainv
    cholesky_df = pd.DataFrame( warped_screens , index=df.index.values, columns=df.columns.values)
    return cholesky_df

def corr_one_vs_all(gene, bf_mat):
    """Correlate one gene essentiality scores vs. all other genes, cheap and fast.
    Parameters
    ----------
    gene - the gene name to extract
    
    bf_mat: dataframe, the N x M matrix of gene essentiality scores
        N = num genes
        M = num samples
    
    Returns
    -------
    pandas.dataframe
      N-1 x 1 frame in which each element is a Pearson correlation coefficient. 
      Index = bf_mat index labels, excluding 'gene'
      Columns = 'gene' parameter
    """

    normed_matrix = cholesky_covariance_warp(bf_mat)

    gene_vector   = normed_matrix.loc[[gene]]
    other_vectors = normed_matrix.drop(gene, axis=0)
    n = gene_vector.shape[1]
    mu_x = gene_vector.mean(1)
    mu_y = other_vectors.mean(1)
    s_x = gene_vector.std(1, ddof=n - 1)
    s_y = other_vectors.std(1, ddof=n - 1)
    cov = np.dot(gene_vector.values, other_vectors.T.values) - n * (mu_x.values * mu_y.values)
    corrmatrix = cov / (s_x.values * s_y.values)
    outframe = pd.DataFrame( index=other_vectors.index.values, columns=[gene], data=corrmatrix.T )
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
        html.Label('Which dataset?'),
        dcc.RadioItems(
            options=[
                {'label': 'Avana/Broad', 'value': 'Avana'},
                {'label': 'Score/Sanger', 'value': 'Score'},
                {'label': 'TKOv3/various', 'value': 'TKOv3'},
            ],
            value='Avana',
            id='dataset',
            persistence=True,
            persistence_type='session',
            labelStyle={'display': 'block'},
        ),
        html.Br(),
        html.Label('Which scoring scheme?'),
        dcc.RadioItems(
            options=[
                {'label': 'Bayes Factor', 'value': 'BF'},
                {'label': 'Z-score', 'value': 'Z'},
            ],
            value='BF',
            id='algo',
            persistence=True,
            persistence_type='local',
            labelStyle={'display': 'block'},

        ),
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
    Input('algo', 'value'),
    Input('gene1', 'value'),
    Input('gene2', 'value'),
    Input('graph_type', 'value'))

def update_figure(dataset, algo, gene1, gene2, graph_type):
    style0 = {'width':'100%','textAlign':'center'}
    style1 = {'width':'100%','textAlign':'left', 'display': 'none'}
    style2 = {'width':'100%','textAlign':'left', 'display': 'none'}
    #gene1 = gene1.upper()
    #gene2 = gene2.upper()
    if (not gene2):
        gene2 = gene1
    my_column = dataset + '_' + algo
    info_area_context_text_content = ''
    display_area_context_text_content = ''
    plot_width = 800
    plot_height = 640
    nowstring = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logstring = nowstring + '\t' + my_column + '\t' + graph_type + '\t' + gene1 + '\t' + gene2 + '\n'
    logfile.write(logstring)

    if (algo=='BF'):
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
        d1 = df[ df.Gene==gene1].sort_values(my_column, ascending=ascend)
        #cells_not_assayed = np.where( np.isnan( d1[my_column]) )[0]
        d1.dropna(axis=0, how='all', inplace=True, subset=[my_column])      # dro   
        d1['idx'] = range(1,d1.shape[0]+1)
        fig = px.scatter(d1, x='idx', y=my_column,
                     hover_name='stripped_cell_line_name',
                     labels = {
                         'idx': 'Cell line rank',
                         my_column: gene1 + ', ' + algo,
                         'primary_disease' : 'Primary Disease'
                     },
                     category_orders = {
                        'primary_disease' : sorted( list( d1.primary_disease.values ) )
                     },
                     color='primary_disease',
                     width=plot_width, height=plot_height)

    elif (graph_type == 'tissue'):
        #
        ########################################
        # "CANCER TYPES" tab 
        #   - hide comparison gene
        #   - scatterplot of query gene essentialty vs comparison gene essentiality
        #   - calculate 
        ########################################
        #        
        d1 = df[ df.Gene==gene1].dropna(axis=0, subset=[my_column])
        #cells_not_assayed = np.where( np.isnan( d1[my_column]) )[0]
        #d1.drop( d1.index.values[ cells_not_assayed ], axis=0, inplace=True)    
        sorted_list = d1.groupby('primary_disease').median()[my_column].sort_values( ascending=ascend )
        disease_rank = pd.DataFrame( index=sorted_list.index.values, columns=['Rank'], data=range(1,len(sorted_list)+1) )
        fig = px.strip(d1, x='primary_disease', y=my_column,
                     hover_name='stripped_cell_line_name',
                     labels = {
                         'median_sort_rank': 'disease type',
                         my_column: gene1 + ', ' + algo + ' (' + dataset + ')',
                         'primary_disease' : 'Primary Disease'
                     },
                     category_orders = {
                        'primary_disease' : list( disease_rank.index.values)
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
        d1 = df[ df.Gene==gene1][[my_column,'stripped_cell_line_name','primary_disease']]
        if (gene1 != gene2):
            d2 = df[ df.Gene==gene2][[my_column,'stripped_cell_line_name']]
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
                        'primary_disease' : sorted( list( d.primary_disease.values ) )
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
        my_correlates = corr_one_vs_all( gene1, essentiality_matrices[my_column] )
        my_correlates[gene1] = round( my_correlates[gene1], 3)
        my_correlates.columns = ['PCC']
        my_correlates.index.name = gene1 
        top_corrs = my_correlates.nlargest(10, 'PCC')
        bot_corrs = my_correlates.nsmallest(10, 'PCC')
        # 
        display_area_context_text_content = html.Div(children=[

            html.Div(id='top_correlates', children = [
                html.H3('Top positive correlates'),
                dt.DataTable(
                    data=top_corrs.reset_index().to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in top_corrs.reset_index().columns],
                    fixed_rows={'headers': True},
                    style_header={'fontSize':'14px', 'font-family':'sans-serif','fontWeight': 'bold','backgroundColor': '#e1e4eb'},
                    style_cell={'minWidth': 100, 'width': 100, 'maxWidth': 100,'fontSize':'12px', 'font-family':'sans-serif'},
                    style_table={'height': 400, 'width':220}
                    )
                ], style={'display':'inline-block', 'padding-right':'5px'}),
            html.Div(id='bot_correlates', children = [
                html.H3('Top negative correlates'),
                dt.DataTable(
                    data=bot_corrs.reset_index().to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in bot_corrs.reset_index().columns],
                    fixed_rows={'headers': True},
                    style_header={'fontSize':'14px', 'font-family':'sans-serif','fontWeight': 'bold','backgroundColor': '#e1e4eb'},
                    style_cell={'minWidth': 100, 'width': 100, 'maxWidth': 100,'fontSize':'12px', 'font-family':'sans-serif'},
                    style_table={'height': 400, 'width':220}
                    )
                ], style={'display':'inline-block','padding-left':'5px'}),
            ])
        style2 = {'width':'100%','textAlign':'left', 'height':450}

    elif (graph_type == 'expr'):
        #
        ########################################
        # "EXPRESSION" tab 
        #   - show comparison gene
        #   - scatterplot of query gene essentialty vs comparison gene expression
        #   - TODO: show correlates?
        ########################################
        #
        d1 = df[ df.Gene==gene1][[my_column,'stripped_cell_line_name']]
        d1.columns = [my_column + '_' + gene1, 'stripped_cell_line_name']
        d2 = df[ df.Gene==gene2][['Expr','stripped_cell_line_name','primary_disease']]
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
                        'primary_disease' : sorted( d.primary_disease.values )
                     },
                     color='primary_disease',
                     width=plot_width, height=plot_height)  
                     #size_max=55)
        style1 = {'width':'100%','textAlign':'left'}
        info_area_context_text_content = 'Comparison gene:'



    elif (graph_type == 'muts'):
        #
        ########################################
        # "MUTATIONS" tab 
        #   - show comparison gene
        #   - strip plot by category of comparison gene
        ########################################
        #
        d1 = df[ df.Gene==gene1][[my_column,'stripped_cell_line_name','primary_disease']]
        d2 = df[ df.Gene==gene2][['LOF','GOF','stripped_cell_line_name']]
        d2['MUT'] = 'WT'
        d2.loc[ d2[ d2.GOF==True].index.values, 'MUT'] = 'GOF'
        d2.loc[ d2[ d2.LOF==True].index.values, 'MUT'] = 'LOF'
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
                        'primary_disease' : sorted( d.primary_disease.values )
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
                html.Li('Score/Sanger: Project Score cell lines screened with the Yusa library (Behan et al release)')]),
                html.Li('TKOv3/various: TKOv3 (Hart et al 2017) screens from several publications. Expression and mutation data not available.'),
            html.P('Scoring scheme', style={'font-weight':'bold'}),
            html.Ul(className='about_scoring', children=[
                html.Li('Bayes Factor: BAGEL2 scoring from Kim & Hart, 2021. Positive scores = essential.'),
                html.Li('Z-score: Gaussian mixture model Z-scores from Lenoir et al 2021 (bioRxiv). Negative scores=essential')]),
            html.P('Query gene', style={'font-weight':'bold'}),
            html.P('the gene selected for fitness profiling'),
            html.P('Comparison gene', style={'font-weight':'bold'}),
            html.P('the query gene\'s fitness is profiled against the comparison gene\'s fitness, expression, or mutation'),
            html.P(''),
            html.P('Download master data file used in this application (Caution: 1.65GB!):'),
            dcc.Link( 'Master data file', 
                href='https://neptune.hart-lab.org/Downloads/master_table_Avana_Score_TKOv3_BF_Zscore_Expr_LOF_GOF_19731genes_1015cells_lineage_disease.txt', 
                style={'font-weight': 'bold'}),
            ])
        style2 = {'width':'100%','textAlign':'left'}
        


    else:
        fig = px.scatter(x=[1], y=[1], width=10, height=10)

    fig.update_layout(transition_duration=500, showlegend=showlegend, font_size=10,)


    return fig, style0, info_area_context_text_content, style1, display_area_context_text_content, style2


if __name__ == '__main__':
    if (DEV_INSTANCE):
        app.run_server(debug=True, port=8052)
    else:
        app.run_server(debug=False)


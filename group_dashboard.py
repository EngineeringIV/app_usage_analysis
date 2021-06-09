import pandas as pd
import os
from utils import *
pd.options.mode.chained_assignment = None

import logging
import pickle
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

tab_style = {
    'height': '36px',
    'fontFamily': 'Sans-Serif',
    'margin-left': 'auto',
    'margin-right': 'auto',
    'padding': '6px',
    'color': '#9E9E9E'
}

tab_selected_style = {
    'height': '36px',
    'fontFamily': 'Sans-Serif',
    'margin-left': 'auto',
    'margin-right': 'auto',
    'padding': '6px'
}

colorscale = [
    'rgb(195, 196, 222)',
    'rgb(144,148,194)',
    'rgb(101,104,168)',
    'rgb(65, 53, 132)'
]



cwd = os.getcwd()
df = pd.read_pickle(cwd+'/usage.pkl')
similarity_df = pd.read_pickle(cwd+'/similarity.pkl')
df_parameter = pd.DataFrame({'Parameter':['App_count', 'Daily_mins']})
df_group = pd.DataFrame({'Group':['Passionate', 'Above average','Ordinary', 'Sleeper']})
head_mapping = df.columns.tolist()
df_display = df.reset_index().sample(50)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    html.H2('Please Select Label, Sublabel, Parameter, and Group'),
    html.Div(children=[
        html.Label('Filters'),
        html.Div(children=[generate_simple_filter('Label', df), generate_empty_filter('Sublabel', df),
                           generate_simple_filter_single('Parameter', df_parameter), generate_simple_filter_single('Group', df_group),
                           generate_simple_filter_single('app_id', df)]

                 ),
        # html.Div(className = "filter", children = [generate_simple_filter(i, df_tran) for i in cat])],

    ], style={'width': '80%', 'display': 'inline-block', 'vertical-align': 'middle'}),
    html.H3(id='my-output'),

    html.H2('Filtered Results', style={'margin-top': 30}),
    dcc.Tabs(id='tabs-example', value='tab-1', children=[
        dcc.Tab(label='Raw', value='tab-1'),
        dcc.Tab(label='Agg (overall)', value='tab-2'),
        dcc.Tab(label='Agg (by label)', value='tab-3'),
        dcc.Tab(label='Group', value='tab-4'),
        dcc.Tab(label='Top 10 Similar Apps (app specific)', value='tab-5')
    ]),
    html.Div(id='tabs-example-content'),
])

@app.callback(
    Output('Sublabel', 'options'),
    [Input('Label', 'value')]
)
def set_cities_options(selected_cdb):
    if any('|' in name for name in selected_cdb):
        return [get_feature_option(df, 'Sublabel')[0]]
    else:
        df_temp = df[df['Label'].isin(list(selected_cdb))]
        return get_feature_option(df_temp, 'Sublabel')


@app.callback(
    Output('Sublabel', 'value'),
    Input('Sublabel', 'options'))
def set_cities_value(available_options):
    return available_options[0]['value']

@app.callback(
    Output('table', 'data'),
    [Input('Label', 'value'),
     Input('Sublabel', 'value'), ])
def update_data(label, sub):
    if '|' in label:
        return df_display.to_dict('records')
    else:
        df2 = df[df['Label'].isin(list(label))]
        if '|' in sub:
            df2 = df2.to_dict("records")
            return df2
        else:
            df2 = df2[df2['Sublabel'].isin(list(sub))]
            df2 = df2.to_dict("records")
            return df2


@app.callback(
    Output('table', 'columns'),
    [Input('Label', 'value'),
     Input('Sublabel', 'value'), ])
def update_column(label, sub):
    if '|' in label:
        return [{"name": i, "id": i} for i in head_mapping]
    else:
        df2 = df[df['Label'].isin(list(label))]
        if '|' in sub:
            df2 = df2.columns.tolist()
            return [{"name": i, "id": i} for i in df2]
        else:
            df2 = df2[df2['Sublabel'].isin(list(sub))]
            df2 = df2.columns.tolist()
            return [{"name": i, "id": i} for i in df2]

@app.callback(
    Output('agg', 'data'),
    [Input('Label', 'value'),
     Input('Sublabel', 'value'),
     Input('Parameter', 'value'),
     Input('Group', 'value'),])
def update_data(label, sub, parameter, group):
    if '|' in label:
        df2 = df
    else:
        df2 = df[df['Label'].isin(list(label))]
        if '|' in sub:
            df2 = df2
        else:
            df2 = df2[df2['Sublabel'].isin(list(sub))]

    df2 = df2.groupby(['user_id'],as_index=False).agg({'app_id':'nunique', 'Label':'nunique', 'Sublabel':'nunique', 'daily_mins':np.mean})
    df2.columns = ['user_id', 'App_count', 'App_label_count', 'App_sublabel_count', 'Daily_mins']
    df2 = df2.to_dict("records")
    return df2


@app.callback(
    Output('agg', 'columns'),
    [Input('Label', 'value'),
     Input('Sublabel', 'value'), ])
def update_column(label, sub):
    if '|' in label:
        df2 = df
    else:
        df2 = df[df['Label'].isin(list(label))]
        if '|' in sub:
            df2 = df2
        else:
            df2 = df2[df2['Sublabel'].isin(list(sub))]
    df2 = df2.groupby(['user_id'], as_index=False).agg(
        {'app_id': 'nunique', 'Label': 'nunique', 'Sublabel': 'nunique', 'daily_mins': np.mean})
    df2.columns = ['user_id', 'App_count', 'App_label_count', 'App_sublabel_count', 'Daily_mins']
    df2 = df2.columns.tolist()
    return [{"name": i, "id": i} for i in df2]

@app.callback(
    Output('agg_label', 'data'),
    [Input('Label', 'value'),
     Input('Sublabel', 'value'),
     Input('Parameter', 'value'),
     Input('Group', 'value'),])
def update_data(label, sub, parameter, group):
    if '|' in label:
        df2 = df
    else:
        df2 = df[df['Label'].isin(list(label))]
        if '|' in sub:
            df2 = df2
        else:
            df2 = df2[df2['Sublabel'].isin(list(sub))]

    df2 = df2.groupby(['user_id', 'Label','Sublabel'],as_index=False).agg({'app_id':'nunique',  'daily_mins':np.mean})
    df2.columns = ['user_id', 'App_label', 'App_sublabel','App_count',  'Daily_mins']
    df2 = df2.to_dict("records")
    return df2


@app.callback(
    Output('agg_label', 'columns'),
    [Input('Label', 'value'),
     Input('Sublabel', 'value'), ])
def update_column(label, sub):
    if '|' in label:
        df2 = df
    else:
        df2 = df[df['Label'].isin(list(label))]
        if '|' in sub:
            df2 = df2
        else:
            df2 = df2[df2['Sublabel'].isin(list(sub))]
    df2 = df2.groupby(['user_id', 'Label','Sublabel'],as_index=False).agg({'app_id':'nunique',  'daily_mins':np.mean})
    df2.columns = ['user_id', 'App_label', 'App_sublabel','App_count',  'Daily_mins']
    df2 = df2.columns.tolist()
    return [{"name": i, "id": i} for i in df2]

@app.callback(
    Output('stats', 'data'),
    [Input('Label', 'value'),
     Input('Sublabel', 'value'),
     Input('Parameter', 'value'),
     Input('Group', 'value'),])
def update_data(label, sub, parameter, group):
    if '|' in label:
        df2 = df
    else:
        df2 = df[df['Label'].isin(list(label))]
        if '|' in sub:
            df2 = df2
        else:
            df2 = df2[df2['Sublabel'].isin(list(sub))]



    df2 = df2.groupby(['user_id'], as_index=False).agg(
        {'app_id': 'nunique', 'Label': 'nunique', 'Sublabel': 'nunique', 'daily_mins': np.mean})
    df2.columns = ['user_id', 'App_count', 'App_label_count', 'App_sublabel_count', 'Daily_mins']
    threshold = list(df2[parameter].describe())
    if group == 'Passionate':
        df2 = df2[df2[parameter]>threshold[6]]
    elif group == 'Above average':
        df2 = df2[(df2[parameter]<=threshold[6])&(df2[parameter]>threshold[5])]
    elif group == 'Ordinary':
        df2 = df2[(df2[parameter] <= threshold[5]) & (df2[parameter] > threshold[4])]
    else:
        df2 = df2[(df2[parameter] == threshold[7])]

    df2 = df2.to_dict("records")
    return df2


@app.callback(
    Output('stats', 'columns'),
    [Input('Label', 'value'),
     Input('Sublabel', 'value'),
     Input('Parameter', 'value'),
     Input('Group', 'value'), ])
def update_column(label, sub, parameter, group):
    if '|' in label:
        df2 = df
    else:
        df2 = df[df['Label'].isin(list(label))]
        if '|' in sub:
            df2 = df2
        else:
            df2 = df2[df2['Sublabel'].isin(list(sub))]
    df2 = df2.groupby(['user_id'], as_index=False).agg(
        {'app_id': 'nunique', 'Label': 'nunique', 'Sublabel': 'nunique', 'daily_mins': np.mean})
    df2.columns = ['user_id', 'App_count', 'App_label_count', 'App_sublabel_count', 'Daily_mins']
    threshold = list(df2[parameter].describe())
    if group == 'Passionate':
        df2 = df2[df2[parameter]>threshold[6]]
    elif group == 'Above average':
        df2 = df2[(df2[parameter]<=threshold[6])&(df2[parameter]>threshold[5])]
    elif group == 'Ordinary':
        df2 = df2[(df2[parameter] <= threshold[5]) & (df2[parameter] > threshold[4])]
    else:
        df2 = df2[(df2[parameter] == threshold[7])]
    df2 = df2.columns.tolist()
    return [{"name": i, "id": i} for i in df2]

@app.callback(
    Output('my-output', 'children'),
    Input('app_id', 'value')
)
def update_output_div(id):

    name = df[df['app_id'] == id]
    name = name['app_name_full'].iloc[0]
    return 'App Name: {}'.format(name)


@app.callback(
    Output('similar', 'data'),
    [Input('app_id', 'value'),])
def update_data(id):
    df2 = find_topk(id, similarity_df, 10)
    df2 = df2.merge(df[['app_id', 'app_name_full', 'Topic', 'Label', 'Sublabel']], on='app_id',
                          how='left').drop_duplicates()
    df2 = df2.to_dict("records")
    return df2


@app.callback(
    Output('similar', 'columns'),
    [Input('app_id', 'value'), ])
def update_column(id):
    df2 = find_topk(id, similarity_df, 10)
    df2 = df2.merge(df[['app_id', 'app_name_full', 'Topic', 'Label', 'Sublabel']], on='app_id',
                    how='left').drop_duplicates()
    df2 = df2.columns.tolist()
    return [{"name": i, "id": i} for i in df2]


@app.callback(Output('tabs-example-content', 'children'),
              Input('tabs-example', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            dcc.Loading(
                type="graph",
                children=html.Div(
                    generate_table(item_id='table', data=df_display.to_dict('records'), head_mapping=head_mapping))),

        ], style={'margin-left': 20, 'margin-right': 20, 'margin-top': 0, 'margin-bottom': 50})

    elif tab == 'tab-2':
        return html.Div([
            dcc.Loading(
                type="graph",
                children=html.Div(
                    generate_table(item_id='agg', data=df_display.to_dict('records'), head_mapping=head_mapping))),

        ], style={'margin-left': 20, 'margin-right': 20, 'margin-top': 0, 'margin-bottom': 50})

    elif tab == 'tab-3':
        return html.Div([
            dcc.Loading(
                type="graph",
                children=html.Div(
                    generate_table(item_id='agg_label', data=df_display.to_dict('records'), head_mapping=head_mapping))),

        ], style={'margin-left': 20, 'margin-right': 20, 'margin-top': 0, 'margin-bottom': 50})

    elif tab == 'tab-4':
        return html.Div([
            dcc.Loading(
                type="graph",
                children=html.Div(
                    generate_table(item_id='stats', data=df_display.to_dict('records'), head_mapping=head_mapping))),

        ], style={'margin-left': 20, 'margin-right': 20, 'margin-top': 0, 'margin-bottom': 50})

    elif tab == 'tab-5':
        return html.Div([
            dcc.Loading(
                type="graph",
                children=html.Div(
                    generate_table(item_id='similar', data=df_display.to_dict('records'), head_mapping=head_mapping))),

        ], style={'margin-left': 20, 'margin-right': 20, 'margin-top': 0, 'margin-bottom': 50})




if __name__ == '__main__':
    app.run_server(debug=False)
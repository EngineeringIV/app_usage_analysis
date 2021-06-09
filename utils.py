import pandas as pd
import numpy as np
import os
from nltk.stem import SnowballStemmer
import gensim
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation, NMF, PCA
from sklearn.preprocessing import normalize, Normalizer
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
from dash_table import DataTable

#Reassign game-related topics for better classification
def reassign_topic(x):
    if x.Topic == 0 or x.Topic==8 or x.Topic==11:
        if not 'game' in x.stemmed:
            return 20
    elif x.Topic == 3 or x.Topic ==1:
        if 'game' in x.stemmed:
            return 0

    return x.Topic


#Stemming the English words to its root form
def stemming(text):
    stemmer = SnowballStemmer("english")
    return stemmer.stem(text)

#Tokenize and stemming text
def preprocess(text):
    '''Takes in a text as a string, returns a single string with tokenized and stemmed words

        Parameters:
            text (str): A text

        Returns:
            (str): Tokenized and stemmed words

    '''
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            res = stemming(token.lower())
            if res not in result:
                result.append(res)

    return " ".join(result)

#Get the top k similar apps from a series of app similarity score
def get_similar_apps(a,topk):
    '''Takes in a series of app similarity score and an integer topk, returns a Series of top k similar apps

        Parameters:
            a (Series): app similarity score
            topk (int): interested number of similar apps

        Returns:
            (Series): top k similar apps
    '''
    # [0] must be itself
    similar_ones = a.sort_values(ascending=False)[1:topk+1].index.values
    return pd.Series(similar_ones,index = ["similar{}".format(i) for i in range(1,topk+1)])

# Add app label and sublabel based on topic number
def add_topic_label(df_temp, category_lut):
    df_temp['Label'] = df_temp['Topic'].apply(lambda x:category_lut.get('Label').get(x))
    df_temp['Sublabel'] = df_temp['Topic'].apply(lambda x: category_lut.get('Sublabel').get(x))
    return df_temp

# Adjust topics
def adjust_topic(df_temp):
    df_temp['Topic'] = df_temp.apply(lambda x:reassign_topic(x), axis=1)
    df_nlp = df_temp[df_temp['Topic'] != 20]
    return df_nlp

# Find the top k similar apps for app
def find_topk(app, similarity, k=1):
    '''Takes in an app_id, similarity dataframe, and integer k, returns a dataframe of top k similar apps

        Parameters:
            app (int): app_id
            similarity (DataFrame): similarity score matrix
            k (int): number of similar apps

        Returns:
            (DataFrame): top k similar apps
    '''
    df = similarity.loc[app].sort_values(ascending=False)[1:k + 1].reset_index()
    df = df.rename(columns={app: 'Similarity'})
    return df

# Calculate the similarity distance matrix and get a dataframe of apps with top k similar apps
def calculate_similarity(df_usage, topk):
    '''Takes in a usage dataframe and integer, returns a similarity distance matrix dataframe and a similar apps dataframe

        Parameters:
            df_usage (DataFrame): app usage dataframe
            topk (int): number of similar apps

        Returns:
            similarity_df: (DataFrame): similarity distance matrix
            top_k_similar_apps: dataframe with top k similar apps for every app in the df_usage dataframe
    '''
    # Calculate average usage time for each user for each app
    app_user = df_usage.groupby(['app_id', 'user_id'])['daily_mins'].mean().unstack(fill_value=0)
    #app_user = app_user.T
    app_user = (app_user > 0).astype(int)

    # Create the similarity score matrix and dataframe
    app_user_norm = normalize(app_user, axis=1)
    similarity = np.dot(app_user_norm, app_user_norm.T)
    similarity_df = pd.DataFrame(similarity, index=app_user.index, columns=app_user.index)

    # Get a dataframe of top k simialr apps for every app
    top_k_similar_apps = similarity_df.apply(get_similar_apps, topk=topk, axis=1)

    return similarity_df, top_k_similar_apps

# Assign app topic using similarity score matrix (simplified collaborative filtering)
def assign_topic_similarity(app_similarity, app_lut,topk):
    '''Takes in a similar app dataframe, app-to-topic lookup table, and an integer, returns a dataframe with app_id and topic

        Parameters:
            app_similarity (DataFrame): similar app dataframe
            app_lut (Dictionary): app-to-topic lookup table
            topk (int): number of similar apps

        Returns:
            df_temp (DataFrame): a dataframe with a list of apps that are not originally in the app-to-topic lookup table.
    '''

    # Label the app based on topic
    df_temp = app_similarity.copy()
    for i in range(topk):
        df_temp['similar'+str(i+1)] = df_temp['similar'+str(i+1)].apply(lambda x:\
                                                app_lut.get('Topic').get(x) if x in app_lut.get('Topic').keys() else 20)

    # Extract the most common app label among the top k similar apps and assign it to app
    df_mode = df_temp.mode(axis=1)
    df_mode['Topic'] = df_mode.apply(lambda x: int(x.iloc[0]) if pd.isna(x.iloc[1]) else int(20), axis=1)

    # Extract and return apps that are not in the original app-to-topic lookup table
    df_temp = df_temp.join(df_mode['Topic'])
    df_temp = df_temp.reset_index()
    df_temp = df_temp[(~(df_temp['app_id'].isin(app_lut.get('Topic').keys()))) & (df_temp['Topic'] != 20)]

    return df_temp[['app_id','Topic']]


# Assign app topic using NLP topic modeling (NMF)
def assign_topic_nmf(df_nlp,tfidf_vect, nmf,fit):
    '''Takes in a dataframe, tfidf vectorizer, nmf model, and boolean, returns a dataframe of apps and topic, tdif vectorizer, and nmf model

        Parameters:
            df_nlp (DatFrame): a dataframe with tokenized and stemmed words as well as app_id
            tfidf_vect (Object): a sklearn TfidfVectorizer object
            nmf (Object): a sklearn NMF object
            fit (bool): a boolean to determine if the tfidf_vectorizer and nmf object need to be fitted

        Returns:
            df_nlp (DatFrame): a dataframe with tokenized and stemmed words, app_id, and app topic
            tfidf_vect (Object): a sklearn TfidfVectorizer object
            nmf (Object): a sklearn NMF object
    '''
    if fit:
        # Fit and transform the TfidfVectorizer and NMF with input dataframe if fit is True.
        tfidf = tfidf_vect.fit(df_nlp['stemmed'].values)
        doc_term_matrix = tfidf.transform(df_nlp['stemmed'].values)
        nmf.fit(doc_term_matrix)

        # Print the top 10 words for each topic
        for i, topic in enumerate(nmf.components_):
            print(f'Top 10 words for topic #{i}:')
            print([tfidf_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
            print('\n')
    else:
        # Transform the input dataframe
        doc_term_matrix = tfidf_vect.transform(df_nlp['stemmed'].values)
        tfidf = tfidf_vect

    # Transform the input dataframe
    topic_values = nmf.transform(doc_term_matrix)

    # Assign and adjust topics based on transformed results
    df_nlp['Topic'] = topic_values.argmax(axis=1)
    df_nlp = df_nlp.sort_values(by='app_id')
    df_nlp = adjust_topic(df_nlp)
    return df_nlp, tfidf,nmf

# Assign app topic using NLP topic modeling (LDA)
def assign_topic_lda(df_nlp,count_vect, lda,fit):
    '''Takes in a dataframe, count vectorizer, lda model, and boolean, returns a dataframe of apps and topic, count vectorizer, lda model

        Parameters:
            df_nlp (DatFrame): a dataframe with tokenized and stemmed words as well as app_id
            count_vect (Object): a sklearn CountVectorizer object
            lda (Object): a sklearn LDA object
            fit (bool): a boolean to determine if the count_vectorizer and lda object need to be fitted

        Returns:
            df_nlp (DatFrame): a dataframe with tokenized and stemmed words, app_id, and app topic
            count_vect (Object): a sklearn CountVectorizer object
            lda (Object): a sklearn LDA object
    '''
    if fit:
        # Fit and transform the CountVectorizer and LDA with input dataframe if fit is True.
        doc_term_matrix = count_vect.fit_transform(df_nlp['stemmed'].values)

        vec = CountVectorizer().fit(df_nlp['stemmed'].values)
        bow = vec.transform(df_nlp['stemmed'].values)
        sum_words = bow.sum(axis=0)

        # Print word frequencies
        word_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        word_freq = sorted(word_freq, key=lambda x: x[1], reverse=True)
        print(word_freq)

        # Print top 10 words in each topic
        lda.fit(doc_term_matrix)
        for i, topic in enumerate(lda.components_):
            print(f'Top 10 words for topic #{i}:')
            print([count_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
            print('\n')
    else:
        # Transform the input dataframe
        doc_term_matrix = count_vect.transform(df_nlp['stemmed'].values)

    # Transform the input dataframe
    topic_values = lda.transform(doc_term_matrix)

    # Assign topics based on transformed results
    df_nlp['Topic'] = topic_values.argmax(axis=1)
    return df_nlp, count_vect, lda

# Unsupervised clustering with Kmeans
def create_cluster(df_usage, df_app,label, cluster_mode, n_clusters):
    '''Takes in a dataframe of app usage, dataframe of app info, list of strings, boolean, and number of clusters, return a dataframe of clusters

        Parameters:
            df_usage (DataFrame): dataframe of app usage
            df_app (DataFrame): dataframe of app information
            label (list(str)): a list of strings that contain the desired label(s) to be filtered
            cluster_mode (str): takes a value of 'app' or 'user' to determine cluster mode
            n_clusters (int): number of clusters

        Returns:
            df (DataFrame): data frame of clusters
    '''
    # Extract data based on specified labels and extract a list of app ids
    if 'All' in label:
        df_temp = df_usage
        temp_id = pd.DataFrame(df_temp['app_id'].unique())
        temp_id.columns = {'app_id'}

    else:
        df_temp = df_usage[df_usage['Label'].isin(label)]
        temp_id = pd.DataFrame(df_temp['app_id'].unique()).rename(columns={0: 'app_id'}).sort_values(by='app_id')
        #temp_id.to_csv(label + '_app_id.csv')

    # Create a app_id - app_name table
    temp_id = temp_id.merge(df_app[['app_id', 'app_name']], on='app_id', how='left')

    # Define the feature to be passed into the pipeline
    feature = df_temp.groupby(['app_id', 'user_id'])['daily_mins'].mean().unstack(fill_value=0)

    # Transpose the feature if clustering user
    if cluster_mode == 'user':
        feature = feature.T

    # Create feature matrix
    feature_matrix = feature.to_numpy()

    ## Plot silhouette coefficient to determine optimum number of clusters
    # plot_silhouette(feature, feature_matrix,range(2,30))

    # Define objects and pipeline
    normalizer = Normalizer()
    pca = PCA()
    kmeans = KMeans(n_clusters=n_clusters, random_state=15)
    pipeline = make_pipeline(normalizer, pca, kmeans)
    pipeline.fit(feature_matrix)
    label = pipeline.predict(feature_matrix)

    # Assign cluster numbers with app_ids
    if cluster_mode == 'user':
        df = pd.DataFrame({'user_id': feature.index, 'label': label})
        for i in range(n_clusters):
            print(i, '\t', list(df[df['label'] == i]['user_id']))
    else:
        df = pd.DataFrame({'app_id': temp_id['app_id'].to_list(), 'label': label})
        df = pd.merge(left=df, right=temp_id, on='app_id', how='left')
        for i in range(n_clusters):
            print(i, '\t', list(df[df['label'] == i]['app_name']))

    return df


# Plot silhouette coefficient plot to determine optimum number of clusters
def plot_silhouette(feature, feature_matrix,clusters):
    inertias=[]
    silhouettes=[]

    for n_clusters in clusters:
        normalizer = Normalizer()
        pca = PCA()
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_jobs=-1)
        pipeline = make_pipeline(normalizer, pca, kmeans)
        pipeline.fit(feature_matrix)
        label = pipeline.predict(feature_matrix)

        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(feature, label))

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
    ax[0].plot(clusters, inertias, 'o-', label='Sum of Squared Distances')
    ax[0].grid(True)
    ax[1].plot(clusters, silhouettes, 'o-', label='Silhouette Coefficient')
    ax[1].grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

# Post process results
def post_process_result(df_usage, similarity_df):
    origin = find_topk(133,similarity_df,10)
    origin = origin.merge(df_usage[['app_id','app_name_full','Topic','Label','Sublabel']], on='app_id',how='left').drop_duplicates()
    steam = find_topk(127, similarity_df, 10)
    steam = steam.merge(df_usage[['app_id','app_name_full','Topic','Label','Sublabel']], on='app_id', how='left').drop_duplicates()
    blizzard = find_topk(207, similarity_df, 10)
    blizzard = blizzard.merge(df_usage[['app_id','app_name_full','Topic','Label','Sublabel']], on='app_id', how='left').drop_duplicates()
    twitch = find_topk(161, similarity_df, 10)
    twitch = twitch.merge(df_usage[['app_id','app_name_full','Topic','Label','Sublabel']], on='app_id', how='left').drop_duplicates()
    discord = find_topk(202, similarity_df, 10)
    discord = discord.merge(df_usage[['app_id','app_name_full','Topic','Label','Sublabel']], on='app_id', how='left').drop_duplicates()

    #df_usage.to_pickle('usage.pkl')
    #similarity_df.to_pickle('similarity.pkl')
    df_user_info = df_usage.groupby(['user_id','Label'],as_index=False).agg({'app_id':'nunique', 'daily_mins':[np.mean,'max']})
    df_user_info.columns=['user_id', 'App_category', 'App_count', 'Daily_mean', 'Daily_max']
    df_user_info_overall = df_usage.groupby(['user_id'],as_index=False).agg({'app_id':'nunique', 'Label':'nunique'})
    df_user_info_overall.columns=['user_id', 'App_count', 'category_count']

    df_game = df_user_info[df_user_info['App_category'] == 'Game']
    game_stats = df_game['App_count'].describe()


    print('done')

# Read and preprocess data
def read_data(cwd,app_name,usage_name,category_name):
    df_app = pd.read_pickle(cwd + app_name)
    df_usage = pd.read_pickle(cwd + usage_name)
    # df_usage.to_pickle('user_app_usage.pkl')
    # df_app.to_pickle('app_information.pkl')
    df_usage = df_usage.merge(df_app, on='app_id', how='left', suffixes=('', '_full'))
    #df_nlp = df_usage[['app_id', 'description']].drop_duplicates().dropna()
    df_nlp = df_app[['app_id', 'description']].drop_duplicates().dropna()
    df_nlp['stemmed'] = df_nlp['description'].apply(preprocess)
    df_nlp = df_nlp.reset_index().drop(['index'], axis=1)
    category_lut = pd.read_csv(cwd + category_name).set_index('Topic').to_dict()

    df_no_desc = df_usage[['app_id', 'description', 'app_name_full']].drop_duplicates()
    df_no_desc = df_no_desc[df_no_desc['description'].isna()]
    df_no_desc = df_no_desc.drop('description', axis=1).dropna()
    df_no_desc['stemmed'] = df_no_desc['app_name_full'].apply(preprocess).dropna()
    df_no_desc = df_no_desc[~df_no_desc['stemmed'].isna()]
    return df_app, df_usage, df_nlp, category_lut, df_no_desc

# Generate dash filter with multiple selection turned on
def generate_simple_filter(feature, df):
    feature_normal = feature
    return \
        html.Div([
            html.Label(feature),
            dcc.Dropdown(
                id=feature,
                options=get_feature_option(df, feature_normal),
                # value=df_tran[feature_normal][0],
                value='|',
                multi=True
            )
        ], className='four columns', style={'width': '15%'})

# Generate dash filter with multiple selection turned off
def generate_simple_filter_single(feature, df):
    feature_normal = feature
    return \
        html.Div([
            html.Label(feature),
            dcc.Dropdown(
                id=feature,
                options=[{'label': i, 'value': i} for i in df[feature].unique()],
                # value=df_tran[feature_normal][0],
                value=df[feature][0],
            )
        ], className='four columns', style={'width': '15%'})

# Generate empty dash filter
def generate_empty_filter(feature, df):
    return \
        html.Div([
            html.Label(feature),
            dcc.Dropdown(
                id=feature,
                multi=True
            )
        ], className='four columns', style={'width': '15%'})

# Get the options for specified features
def get_feature_option(df, feature):
    tag_list = df[feature].dropna().unique()
    option_list = [{'label': i, 'value': i} for i in tag_list]
    option_list.insert(0, ({'label': 'All', 'value': '|'}))

    return option_list

# Generate dash DataTable
def generate_table(item_id, data, head_mapping):
    return DataTable(
        id=item_id,
        columns=[{"name": i, "id": i} for i in head_mapping],
        editable=True,
        data=data,
        sort_action="native",
        filter_action="native",
        # sort_mode="multi",
        # column_selectable="single",
        # row_selectable="multi",
        # row_deletable=deletable,
        style_table={'overflowX': 'scroll',
                     'overflowY': 'scroll',
                     'height': 550,
                     'padding-left': 20},

        style_data={'whiteSpace': 'normal',
                    'height': 'auto',
                    'font-size': '12px',
                    'font-family': '-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,"Noto Sans",sans-serif,"Apple Color Emoji","Segoe UI Emoji","Segoe UI Symbol","Noto Color Emoji"'},

        style_cell={
            'height': 'auto',
            'minWidth': '0px',
            'maxWidth': '480px',
            'whiteSpace': 'normal',
            'textAlign': 'center',
            'border': '1px solid grey'
        },

        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(227,242,253)'
            }
        ],

        style_header={

            'backgroundColor': '#757575',
            'fontWeight': 'bold',
            'font-family': '-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,"Noto Sans",sans-serif,"Apple Color Emoji","Segoe UI Emoji","Segoe UI Symbol","Noto Color Emoji"',
            'font-size': '14px',
            'padding-top': '10px',
            'padding-bottom': '10px'
        },
        page_current=0,
        page_size=20,
        export_format="csv"
        # page_action='custom'
    )
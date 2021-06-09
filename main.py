import os
from utils import *

import warnings
warnings.simplefilter('ignore')


if __name__ == '__main__':
    #Read and preprocess app information and user usage dataset
    cwd = os.getcwd()
    df_app, df_usage, df_nlp, category_lut, df_no_desc = read_data(cwd=cwd+'/Data',
                                                                   app_name='/app_information.pkl',
                                                                   usage_name='/user_app_usage.pkl',
                                                                   category_name='/category_lut_new.csv')

    # NLP topic modeling with Latent Dirichlet Allocation for apps with description
    # count_vect = CountVectorizer(min_df=3, stop_words='english', ngram_range=(1, 2))
    # lda = LatentDirichletAllocation(n_components=15, random_state=8)
    # df_nlp, tfidf, nmf = assign_topic_lda(df_nlp, count_vect, lda, True)

    # NLP topic modeling with Non-negative Matrix Factorization for apps with description
    tfidf = TfidfVectorizer(min_df=3, stop_words=ENGLISH_STOP_WORDS, ngram_range=(1,2))
    nmf = NMF(n_components=15, random_state=6)
    df_nlp,tfidf,nmf = assign_topic_nmf(df_nlp,tfidf,nmf,True)

    # NLP topic modeling with Non-negative Matrix Factorization for apps without description
    df_no_desc,tfidf,nmf = assign_topic_nmf(df_no_desc,tfidf,nmf,False)

    # Merge dataframes together and create an app-to-label look up table
    df_nlp = df_nlp[['app_id','Topic']].append(df_no_desc[['app_id','Topic']])
    df_nlp = add_topic_label(df_nlp,category_lut)
    app_lut = df_nlp[['app_id','Topic','Label', 'Sublabel']].set_index('app_id').to_dict()

    # Similarity calculation
    topk = 10
    similarity_df, top_k_similar_apps = calculate_similarity(df_usage, topk)
    app_similarity_topic = assign_topic_similarity(top_k_similar_apps,app_lut,topk)
    df_nlp = df_nlp.append(app_similarity_topic)

    # Merge the labeled app information with the usage data to create a complete dataset for later use
    df_usage = df_usage.merge(df_nlp,on='app_id',how='left').fillna(20)
    df_usage = add_topic_label(df_usage, category_lut)

    # App/user clustering with KMeans
    df_cluster = create_cluster(df_usage=df_usage, df_app=df_app,label=['Game'], cluster_mode='app', n_clusters=6)

    #Post-process results
    post_process_result(df_usage, similarity_df)
    game_list = df_nlp[df_nlp['Label'] == 'Game'].merge(df_app[['app_id', 'app_name', 'description']], on='app_id',
                                                        how='left') \
        .sort_values(by='app_id')
    game_list.to_csv('game_id.csv')





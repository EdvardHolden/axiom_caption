# For EDA
from sklearn.decomposition import LatentDirichletAllocation

cvectorizer = CountVectorizer(min_df=2,
                              max_features=180020)
cvz = cvectorizer.fit_transform(text)


no_categories = no_clusters
lda_model = LatentDirichletAllocation(n_components=no_categories,
                                      learning_method='online',
                                      max_iter=20,
                                      random_state=42)
X_topics = lda_model.fit_transform(cvz)

n_top_words = 5
topic_summaries = []
topic_word = lda_model.components_  # get the topic words
vocab = cvectorizer.get_feature_names()

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))
    print('Topic {}: {}'.format(i, ' | '.join(topic_words)))



def cluster_companies_by_main_topic(lda, num_topics, corpus, companies):
	"""Group companies by their most relevant topic.

	Parameters
	----------
	lda : LDA Model
		The fitted LDA model.
	num_topics : int
		The number of topics used in the LDA model.
	corpus : list
		The document corpus used to train the lda.
	companies : list
		The list of companies associated with each text in the corpus.

	Returns
	-------
	dict
		An dictionary indexed by the topic integer where values correspond the
		list companies for which this is the most relevant topic.
	"""

	cluster = {}
	for i in range(num_topics):
		cluster[i] = []
	for idx, company in enumerate(companies):
		mod_pred = lda[corpus[idx]]
		weights = [w[1] for w in mod_pred]
		idx_max = weights.index(max(weights))
		max_topic = mod_pred[idx_max][0]
		cluster[max_topic].append(company)
	return cluster
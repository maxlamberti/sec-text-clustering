import re
import requests
from tqdm import tqdm
from nltk import pos_tag
from bs4 import BeautifulSoup
from gensim.models.phrases import Phraser
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer


def convert_attr_to_dict(attr):
	"""Convert HTML text attributes to python dict.

	Parameters
	----------
	attr : str
		Example: "font-family:Helvetica,sans-serif;font-size:11pt;font-weight:bold;"

	Returns
	-------
	dict
		Example {"font-family": "Helvetica, sans-serif", ...}
	"""

	result = dict()
	attr = attr.split(';')
	attrlist = [a.split(':') for a in attr]
	for pair in attrlist:
		if len(pair) == 2:
			key = pair[0]
			value = pair[1]
			result[key] = value

	return result


def get_10k_text(url):
	"""Scrape relevant text from a 10-k document in the SEC Edgar database.

	Parameters
	----------
	url : str
		URL pointing to the 10-k document in HTML format.

	Returns
	-------
	headers : list
		A list of the headers of the recorded sections.
	text : list
		A list of recorded sections from the relevant parts of the document.
	"""

	rawdoc = BeautifulSoup(requests.get(url).content, 'html.parser')

	headers = []
	text = []
	start_recording = False  # record headers between business & risk factors section
	for data in rawdoc.html.body:
		try:
			attributes = convert_attr_to_dict(data.find('font')['style'])
			is_header = attributes.get('font-weight') == 'bold'  # recognize bold text as header
			if is_header:
				if 'business' in data.text.lower():  # start of relevant section
					start_recording = True
				if 'risk factors' in data.text.lower():  # end of relevant section
					break
				if start_recording:
					headers.append(data.text)
			else:  # assume text is paragraph section
				if start_recording:  # record text
					text.append(data.text)
		except:  # cheeky too broad catch all -> we are going to filter out poorly scraped data later
			continue

	return headers, text


def scrape_edgar(urls):
	documents = {}
	section_headers = {}
	for company, url in tqdm(urls.items()):
		headers, doc = get_10k_text(url)
		if doc:  # only keep company if scraping was successful
			documents[company] = doc
			section_headers[company] = headers
			for header in headers:  # correct missidentified headers
				if len(header) > 100:
					documents[company].append(header)
	return documents


def format_text(text):
	"""Remove special characters, too many spaces, tokenize and add POS tagging.

	Parameters
	----------
	text : list
		List of raw scraped text from one 10-k document.

	Returns
	-------
	list
		A list of of POS tagged word tokens.
	"""

	text = ' '.join(text).lower()
	text = re.sub(r"[^a-zA-Z.?!]", " ", text)
	text = re.sub(r' +', ' ', text)
	text = word_tokenize(text)
	text = pos_tag(text)

	return text


def map_pos_tag(pos):
	mappings = {'NN': wn.NOUN, 'JJ': wn.ADJ, 'VB': wn.VERB, 'RB': wn.ADV}
	pos = pos[:2]
	if pos in mappings:
		pos = mappings[pos]
	else:
		pos = wn.NOUN
	return pos


def lem_and_stem(text, stopwords):
	lemmatizer = WordNetLemmatizer()
	stemmer = PorterStemmer()
	processed_text = []
	for token, pos in text:
		pos = map_pos_tag(pos)
		if not (pos == wn.NOUN):
			continue
		if token not in stopwords and len(token) > 3:
			processed_token = stemmer.stem(lemmatizer.lemmatize(token, pos=pos))
			if processed_token not in stopwords:
				processed_text.append(processed_token)
	return processed_text


def add_bigrams(text):

	bigram = Phraser(text, min_count=20)  # min freq of 20
	for idx in range(len(text)):
		for token in bigram[text[idx]]:
			if '_' in token:
				# Token is a bigram, add to document.
				text[idx].append(token)

	return text

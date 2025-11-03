"""
Module to extract stylometric features from speech transcript text.

Some of the features are adapted from Strom (2021) (https://github.com/eivistr/pan21-style-change-detection-stacking-ensemble),
...who adapted from Zuo et al. (2019) (https://github.com/chzuo/PAN_2019), 
...who adapted from Zlatkova et al. (2018) (https://github.com/machinelearning-su/style-change-detection)
...who most likely adapted from vast earlier works in stylometry.

Other features are adapated from Altakrori et al. (2021) (https://github.com/malikaltakrori/Topic-Confusion-for-authorship-attribution-EMNLP-2021-)

Other features were created/adapted by the authors specifically for this domain.
"""

import os
import json
from collections import Counter
import textstat


############## ------------- CHARACTER properties ------------- ##############

### Get punctuation mark frequencies
def punctuation_freqs(tokens): #use tokens not str_doc so doesn't count multi-marks (e.g. '...') separately
	punct_marks = [ '.', '?', '!', ',', ';', ':', '-', '--', '---', '..', '...', '(', ')', '[', ']', '\'', '"', '`'] #18 total
	punct_counts = [tokens.count(punct) for punct in punct_marks]
	return punct_counts 


############## ------------- TOKEN/WORD/POS properties ------------- ##############

### Get POS tag frequencies and word/token properties 
def word_pos_freqs(stanza_doc, tokens): 
	token_counts = Counter(tokens) 
	postag_list = [0] * 16
	word_lengths = []
	word_count = 0
	long_words = 0
	short_words = 0
	capitalized_words = 0
	wordprop_list = [0] * 3
	wordratio_list = [0] * 4

	for sent in stanza_doc.sentences: #loop through this here to avoid looping through both a pos_tag list here and then a words list below (latter b/c need words that haven't been lowercased)
		for token in sent.words:
			if token.upos in ['ADJ']:
				postag_list[0] += 1
			elif token.upos in ['ADP']:
				postag_list[1] += 1
			elif token.upos in ['ADV']:
				postag_list[2] += 1
			elif token.upos in ['AUX']:
				postag_list[3] += 1
			elif token.upos in ['CCONJ']:
				postag_list[4] += 1
			elif token.upos in ['DET']:
				postag_list[5] += 1
			elif token.upos in ['INTJ']:
				postag_list[6] += 1
			elif token.upos in ['NOUN', 'PROPN']:
				postag_list[7] += 1
			elif token.upos in ['NUM']:
				postag_list[8] += 1
			elif token.upos in ['PART']:
				postag_list[9] += 1
			elif token.upos in ['PRON']:
				postag_list[10] += 1
			elif token.upos in ['PUNCT']:
				postag_list[11] += 1
			elif token.upos in ['SCONJ']:
				postag_list[12] += 1
			elif token.upos in ['SYM']:
				postag_list[13] += 1
			elif token.upos in ['VERB']:
				postag_list[14] += 1
			elif token.upos in ['X']:
				postag_list[15] += 1
	
			## Word properties (words only, no punct/numbers/symbols)
			if token.upos not in ['NUM','PUNCT','SYM']:
				if len(token.text) >= 8: #Strom uses >20; long words
					long_words += 1
				if len(token.text) < 5: #arbitrary (Strom uses <5, Altakrori uses <4); short words
					short_words += 1 
				elif token.text[0].isupper(): 
					capitalized_words += 1
				word_lengths.append(len(token.text))
				word_count += 1 #word count (W)
	wordprop_list[0] = sum(word_lengths) / word_count # average word length

	## Total num tokens
	num_tokens = len(tokens) #token count (T)
	wordprop_list[1] = num_tokens #tokens not words b/c includes punct
	num_unique_tokens = len(token_counts)
	wordprop_list[2] = num_unique_tokens #also includes punct
	
	## Word ratios (Altakrori et al. 2021 + authors added some)
	wordratio_list[0] = short_words / word_count #ratio of num of short words to num of words (Altakrori used num_tokens instead of word_count)
	wordratio_list[1] = long_words / word_count #ratio of num of long words to num of words (Altakrori used num_tokens instead of word_count)
	wordratio_list[2] = capitalized_words / word_count #ratio of num of capitalized words to num of words (Altakrori did not do this) 
	wordratio_list[3] = num_unique_tokens / num_tokens #ratio of word types to num of tokens (type:token) 

	word_features = wordprop_list + wordratio_list + postag_list
	return word_features 


############## ------------- Other SYNTAX properties ------------- ##############
### Get properties of sentences
def sentence_props(stanza_doc): 
	total_sent_lens = []
	for sent in stanza_doc.sentences:
		num_tokens_sent = len(list(sent.words)) #Stanza doesn't seem to be affected by double spaces; spaCy counts 1 of the spaces in a double space as a token
		total_sent_lens.append(num_tokens_sent)
	num_sentences = len(list(stanza_doc.sentences))
	avg_sent_length = sum(total_sent_lens) / num_sentences #includes punct as part of length
	return num_sentences, avg_sent_length 


def function_words(str_doc, tokens, work_dir):
	## Function words (authors augmented from Strom 2021)
	function_words_aug = os.path.join(work_dir, 'function_words_augmented.json')
	if not os.path.exists(function_words_aug):
		raise FileNotFoundError(f"Required file not found: {function_words_aug}")
	with open(function_words_aug, 'r') as f:
		func_words = json.load(f) #augmented from original (orig = NLTK stopwords + Zlatkova 2018)

	token_counts = Counter(tokens) 
	func_word_feature = []
	for w in func_words['words']:
		if w in token_counts:
			func_word_feature.append(token_counts[w])
		else:
			func_word_feature.append(0) 
	func_phrase_feature = [str_doc.lower().count(p) for p in func_words['phrases']]
	return func_word_feature, func_phrase_feature 


############## ------------- DISCOURSE properties ------------- ##############
### Vocab richness (higher # = higher diversity/richer vocab) 
## Source: [https://gist.github.com/magnusnissel/d9521cb78b9ae0b2c7d6]
def Yules_i(lex_words): #
	word_counts = Counter(lex_words) #dict of counts of lexical items only (no numbers, punct, symbols)
	m1 = sum(word_counts.values())
	m2 = sum([freq ** 2 for freq in word_counts.values()])
	try:
		i = (m1 * m1) / (m2 - m1)
	except ZeroDivisionError:
		i = 0
	#k = 1 / i * 10000 # Yule's k
	# return (k, i)
	return i


### Get readability scores
def readability_features(string_doc):
	textstat_scores = [textstat.flesch_reading_ease(string_doc),
						 textstat.smog_index(string_doc),
						 textstat.flesch_kincaid_grade(string_doc),
						 textstat.coleman_liau_index(string_doc),
						 textstat.automated_readability_index(string_doc),
						 textstat.dale_chall_readability_score(string_doc),
						 textstat.difficult_words(string_doc),
						 textstat.linsear_write_formula(string_doc),
						 textstat.gunning_fog(string_doc)]
	return textstat_scores


### Get hapax legomena/dislegomena per speaker in a side of a call (not based on whole dataset)
def hapax(lex_words): 
	word_counts = Counter(lex_words) 
	hl_per_speaker = 0
	hd_per_speaker = 0
	for word in word_counts:
		if word_counts[word] == 1:
			hl_per_speaker += 1
		elif word_counts[word] == 2:
			hd_per_speaker += 1
	hl_normed = hl_per_speaker / len(lex_words) #normalized
	hd_normed = hd_per_speaker / len(lex_words) #normalized
	return hl_normed, hd_normed 


### Get (augmented) stylistic contraction choices 
def contractions(str_doc, work_dir):  
	## Stylistic choices (authors augmented from Strom 2021)
	comparison_lists = os.path.join(work_dir, 'comparison_lists_augmented.json')
	if not os.path.exists(comparison_lists):
		raise FileNotFoundError(f"Required file not found: {comparison_lists}")
	with open(comparison_lists, 'r') as f:
		comparison_dict = json.load(f) #augmented from original (= NLTK stopwords + Zlatkova 2018)

	comparison_counts = [count_occurence_phrase(comparison_dict['contractions'][0], str_doc), 
						count_occurence_phrase(comparison_dict['contractions'][1], str_doc)] #use text as string b/c otherwise contractions are tokenized into sep. tokens 
	return comparison_counts 


### Helper function for contractions
def count_occurence_phrase(phrase_list, str_doc): 
	num_count = 0
	for phrase in phrase_list:
		num_count += str_doc.lower().count(phrase)
	### NORMED:
	# num_count_normed = num_count / len(tokens) * norm
	return num_count #num_count_normed 


############## ------------- GET ALL STYLOMETRIC FEATURES + NAMES ------------- ##############

def get_stylo_features(preprocessed_texts, work_dir): 
	docs_features = []
	for preprocessed_text in preprocessed_texts:
		string_doc = preprocessed_text['string'] 
		stanza_doc = preprocessed_text['stanza_doc']
		tokens = preprocessed_text['tokens']
		lex_words = preprocessed_text['words']
		
		## Character properties
		punct_counts = punctuation_freqs(tokens) 

		## Token/word and POS properties
		word_features = word_pos_freqs(stanza_doc, tokens) #need to preserve capitalization 
		
		## Other syntax properties
		num_sents, avg_sent_length = sentence_props(stanza_doc) 
		func_word_feature, func_phrase_feature = function_words(string_doc, tokens, work_dir) 

		## Discourse properties
		yules = Yules_i(lex_words)
		textstat_scores = readability_features(string_doc)
		hap_leg, hap_disleg = hapax(lex_words)
		comparison_counts = contractions(string_doc, work_dir) 

		### Collect all features
		features = [punct_counts, word_features, num_sents, avg_sent_length, func_word_feature, 
			  func_phrase_feature, yules, textstat_scores, hap_leg, hap_disleg, comparison_counts]			
		features_flat = []
		for elem in features:
			if type(elem) == list:
				for item in elem:
					features_flat.append(item)
			else:
				features_flat.append(elem)
		docs_features.append(features_flat)
	return docs_features


### NOTE Must keep the same order as in get_stylo_features()
def get_feature_names(work_dir):
	## Character properties
	punct_marks = [ '.', '?', '!', ',', ';', ':', '-', '--', '---', '..', '...', '(', ')', '[', ']', '\'', '"', '`'] 

	## Token/word and POS properties
	word_properties = ['avgwordlength','#totaltokens','#uniqtokens'] 
	word_ratios = ['shortwords:W','longwords:W','capitalized:W','wordtypes:T']
	pos_categories = ['ADJ','ADP','ADV','AUX','CCONJ','DET','INTJ','NOUNs','NUM','PART','PRON','PUNCT','SCONJ','SYM','VERB','X']

	## Other syntax properties
	function_words_aug = os.path.join(work_dir, 'function_words_augmented.json')
	if not os.path.exists(function_words_aug):
		raise FileNotFoundError(f"Required file not found: {function_words_aug}")
	with open(function_words_aug, 'r') as f:
		func_words = json.load(f) 

	## Discourse properties
	textstat_names = ['FleschReadingEase','SMOGindex','Flesch-KincaidGradeLevel','Coleman-LiauIndex','AutomatedReadIndex','Dale-ChallReadScore','DifficultWords','LinsearWriteFormula','GunningfogIndex']
	comparison_names = ['contracted','not contracted']

	### Collect all feature names (order matters)
	feature_names = [punct_marks, word_properties, word_ratios, pos_categories, 'num sents', 
			'avg sent length', func_words['words'], func_words['phrases'], 'Yules_i', textstat_names, 'hapax legomena', 'hapax dislegomena', comparison_names]
	feature_names_flat = []
	for elem in feature_names:
		if type(elem) == list:
			for item in elem:
				feature_names_flat.append(item)
		else:
			feature_names_flat.append(elem)
	return feature_names_flat
"""
StyloSpeaker
Preprocess texts, extract features, train and evaluate classifiers
Some of the code is adapted from Weerasinghe & Greenstadt (2020): https://github.com/janithnw/pan2020_authorship_verification
"""

import sys
import os
import yaml
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from scipy.sparse import hstack, vstack, save_npz, load_npz
import joblib
from joblib import dump
import stanza
# stanza.download('en') #only do once

from stylometric_features import get_stylo_features, get_feature_names


### Get pairs of preprocessed texts and their verification label
def get_pairs_labels(trials):
	## Collect unique call texts to not reprocess duplicates
	unique_calls = {}
	for trial in trials:
		for key in ["call 1", "call 2"]:
			call_lines = trial[key]
			call_str = " ".join(call_lines).replace('  ', ' ')
			if call_str not in unique_calls:
				unique_calls[call_str] = None  #placeholder

	## Stanza preprocessing (avoid redundant processing of duplicate texts)
	unique_texts = list(unique_calls.keys())
	preprocessed_texts = preprocess_texts(unique_texts)
	print('Num of trials:', len(trials))
	print('Num of total texts:', len(trials) * 2) #2 texts per trial
	print('Num of unique texts preprocessed:', len(preprocessed_texts))

	## Map texts back to call IDs
	call_str_to_id = {text: f"call_{i}" for i, text in enumerate(unique_texts)}
	texts = {call_str_to_id[text]: preprocessed_texts[i] for i, text in enumerate(unique_texts)}

	## Generate pairs and labels
	pairs = [] 
	labels = [] 
	for trial in trials:
		c1 = call_str_to_id[" ".join(trial["call 1"]).replace('  ', ' ')]
		c2 = call_str_to_id[" ".join(trial["call 2"]).replace('  ', ' ')]
		pairs.append((c1, c2)) #[('call_0', 'call_1'),...]
		labels.append(trial["label"])
	return pairs, labels, texts


### Preprocess documents to include raw string, tokens, and POS tags
def preprocess_texts(unique_texts):
	nlp = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=True,
						pos_batch_size=500)
	### Stanzafy docs all together for faster processing
	in_docs = [stanza.Document([], text=t) for t in unique_texts] #wrap each w/stanza.Document obj
	stanza_docs = nlp(in_docs) 

	## Extract tokens and POS tags
	preprocessed_texts = []
	for text, stanza_doc in zip(unique_texts, stanza_docs):
		tokens = [] #includes words + punctuation/numbers/symbols (i.e. non-words) 
		lex_words = [] #only lexical items, excludes punct, numbers, and symbols
		pos_tags = []
		for sent in stanza_doc.sentences:
			for token in sent.words:  
				tokens.append(token.text.lower()) 
				if token.upos not in ['NUM','PUNCT','SYM']:
					lex_words.append(token.text.lower())
				if token.upos in ['PUNCT','SYM']:
					pos_tags.append(token.text)
				else:
					pos_tags.append(token.upos)
		preprocessed_texts.append({"string": text, 
								"stanza_doc": stanza_doc, 
								"tokens": tokens, 
								"words": lex_words,
								"pos_tags": ' '.join(pos_tags)})
	return preprocessed_texts


### Stylometric feature extractor
class StylometricFeatures(BaseEstimator, TransformerMixin):
	def __init__(self, work_dir):
		self.wd = work_dir

	def fit(self, X, y=None):
		return self

	def transform(self, X): #X = preprocessed texts
		return np.array(get_stylo_features(X, self.wd)) 

	def get_feature_names_out(self, input_features=None):
		return np.array(get_feature_names(self.wd)) 


## Extract the raw string text from preprocessed docs
def extract_string_text(docs):
	return [doc["string"] for doc in docs]


## Extract the POS tag text from preprocessed docs
def extract_pos_tag_text(docs):
	return [doc["pos_tags"] for doc in docs]


## Build feature matrix for unique docs
def build_doc_feature_matrix(doc_ids, doc_texts, pipeline, fit=False):
	texts = [doc_texts[d] for d in doc_ids]
	if fit:
		X = pipeline.fit_transform(texts)
	else:
		X = pipeline.transform(texts)
	return {doc: X[i] for i, doc in enumerate(doc_ids)}, texts


## Build pairwise features
def build_pairwise_matrix(pairs, doc2features, feat_combos):
	pair_features = []
	for d1, d2 in pairs:
		f1, f2 = doc2features[d1], doc2features[d2]
		diff = f1 - f2
		prod = f1.multiply(f2)
		if feat_combos == 'feats':
			pair_feat = hstack([f1, f2])
		elif feat_combos == 'diff':
			pair_feat = diff
		elif feat_combos == 'diffabs': 
			pair_feat = abs(diff) 
		elif feat_combos == 'featsdiffabs':
			pair_feat = hstack([f1, f2, abs(diff)])
		elif feat_combos == 'diffprod':
			pair_feat = hstack([diff, prod])
		elif feat_combos == 'diffabsprod': 
			pair_feat = hstack([abs(diff), prod])
		if feat_combos == 'featsdiffprod':
			pair_feat = hstack([f1, f2, diff, prod])
		pair_features.append(pair_feat)
	return vstack(pair_features)


def get_feat_names(doc_feature_pipeline, feat_combos):
	### Get feature names
	fitted_char = doc_feature_pipeline.transformer_list[0][1].named_steps["char_tfidf_vec"]
	fitted_token = doc_feature_pipeline.transformer_list[1][1].named_steps["token_tfidf_vec"]
	fitted_pos = doc_feature_pipeline.transformer_list[2][1].named_steps["pos_tfidf_vec"]
	fitted_stylo = doc_feature_pipeline.transformer_list[3][1].named_steps["extract"]

	char_names = fitted_char.get_feature_names_out()
	token_names = fitted_token.get_feature_names_out()
	pos_names_lower = fitted_pos.get_feature_names_out()
	pos_names = [name.upper() for name in pos_names_lower]
	stylo_names = fitted_stylo.get_feature_names_out()

	### Prep all features for feature importance
	if feat_combos == 'feats': #only individual features 
		all_feat_names = [ 
			f"char1_{f}" for f in char_names
		] + [
			f"token1_{f}" for f in token_names
		] + [
			f"pos1_{f}" for f in pos_names
		] + [
			f"stylo1_{f}" for f in stylo_names
		] + [
			f"char2_{f}" for f in char_names
		] + [
			f"token2_{f}" for f in token_names
		] + [
			f"pos2_{f}" for f in pos_names
		] + [
			f"stylo2_{f}" for f in stylo_names
		] 
	elif feat_combos == 'diff': #difference of features b/t sides of trials
		all_feat_names = [f"diff_char_{f}" for f in char_names
		] + [
			f"diff_tok_{f}" for f in token_names
		] + [
			f"diff_pos_{f}" for f in pos_names
		] + [
			f"diff_stylo_{f}" for f in stylo_names
		] 
	elif feat_combos == 'diffabs': #absolute difference of features b/t sides of trials
		all_feat_names = [f"diffabs_char_{f}" for f in char_names
		] + [
			f"diffabs_tok_{f}" for f in token_names
		] + [
			f"diffabs_pos_{f}" for f in pos_names
		] + [
			f"diffabs_stylo_{f}" for f in stylo_names
		] 
	elif feat_combos == 'featsdiffabs': #individual features + absolute difference of features b/t sides of trials
		all_feat_names = [
			f"char1_{f}" for f in char_names
		] + [
			f"token1_{f}" for f in token_names
		] + [
			f"pos1_{f}" for f in pos_names
		] + [
			f"stylo1_{f}" for f in stylo_names
		] + [
			f"char2_{f}" for f in char_names
		] + [
			f"token2_{f}" for f in token_names
		] + [
			f"pos2_{f}" for f in pos_names
		] + [
			f"stylo2_{f}" for f in stylo_names
		] + [
			f"diffabs_char_{f}" for f in char_names
		] + [
			f"diffabs_tok_{f}" for f in token_names
		] + [
			f"diffabs_pos_{f}" for f in pos_names
		] + [
			f"diffabs_stylo_{f}" for f in stylo_names
		] 
	### More complex interactions between features
	elif feat_combos == 'diffprod': #diff and product of features b/t sides of trials
		#tried diffprod_wb but got the same results as diffprod
		all_feat_names = [f"diff_char_{f}" for f in char_names
		] + [
			f"diff_tok_{f}" for f in token_names
		] + [
			f"diff_pos_{f}" for f in pos_names
		] + [
			f"diff_stylo_{f}" for f in stylo_names
		] + [
			f"prod_char_{f}" for f in char_names
		] + [
			f"prod_tok_{f}" for f in token_names
		] + [
			f"prod_pos_{f}" for f in pos_names
		] + [
			f"prod_stylo_{f}" for f in stylo_names
		]
	elif feat_combos == 'diffabsprod': #absolute diff and prod of feats b/t sides of trials
		all_feat_names = [f"diffabs_char_{f}" for f in char_names
		] + [
			f"diffabs_tok_{f}" for f in token_names
		] + [
			f"diffabs_pos_{f}" for f in pos_names
		] + [
			f"diffabs_stylo_{f}" for f in stylo_names
		] + [
			f"prod_char_{f}" for f in char_names
		] + [
			f"prod_tok_{f}" for f in token_names
		] + [
			f"prod_pos_{f}" for f in pos_names
		] + [
			f"prod_stylo_{f}" for f in stylo_names
		]
	elif feat_combos == 'featsdiffprod': #individual feats + diff + prod of feats b/t sides of trials
		all_feat_names = [ 
			f"char1_{f}" for f in char_names
		] + [
			f"token1_{f}" for f in token_names
		] + [
			f"pos1_{f}" for f in pos_names
		] + [
			f"stylo1_{f}" for f in stylo_names
		] + [
			f"char2_{f}" for f in char_names
		] + [
			f"token2_{f}" for f in token_names
		] + [
			f"pos2_{f}" for f in pos_names
		] + [
			f"stylo2_{f}" for f in stylo_names
		] + [
			f"diff_char_{f}" for f in char_names
		] + [
			f"diff_tok_{f}" for f in token_names
		] + [
			f"diff_pos_{f}" for f in pos_names
		] + [
			f"diff_stylo_{f}" for f in stylo_names
		] + [
			f"prod_char_{f}" for f in char_names
		] + [
			f"prod_tok_{f}" for f in token_names
		] + [
			f"prod_pos_{f}" for f in pos_names
		] + [
			f"prod_stylo_{f}" for f in stylo_names
		]
	return all_feat_names


def compute_metrics(clf, X_test, y_test):
	preds = clf.predict(X_test) #accuracy
	probs = clf.predict_proba(X_test)[:, 1] #auc, eer
	
	acc = accuracy_score(y_test, preds)
	print("Test Accuracy:", acc)

	roc_auc = roc_auc_score(y_test, probs)
	print("ROC AUC:", roc_auc)

	## Compute EER
	fpr, tpr, thresholds = roc_curve(y_test, probs)
	fnr = 1 - tpr
	eer_threshold_index = np.nanargmin(np.absolute(fnr - fpr))
	eer = (fpr[eer_threshold_index] + fnr[eer_threshold_index]) / 2
	print("EER:", eer)
	return acc, roc_auc, eer



####----------------------------------------MAIN---------------------------------------------####
def main(cfg):
	### Get config parameters
	work_dir = cfg['work_dir']
	out_dir = cfg['out_dir']
	trials_dir = cfg['trials_dir']
	encodings = cfg['encodings'] 
	levels = cfg['levels']
	feat_combos = cfg['feat_combos']
	char_ngram_range = tuple(cfg['char_ngram_range'])
	tok_ngram_range = tuple(cfg['tok_ngram_range'])
	pos_ngram_range = tuple(cfg['pos_ngram_range'])
	which_clf = cfg['which_clf']

	### Build the feature extraction pipeline
	doc_feature_pipeline = FeatureUnion([
		("char_tfidf", Pipeline([ 
			("get_string", FunctionTransformer(extract_string_text, validate=False)),
			("char_tfidf_vec", TfidfVectorizer(analyzer="char", ngram_range=char_ngram_range, 
					lowercase=True, min_df=0.1, norm='l2', max_features=2000)) 
		])),
		("token_tfidf", Pipeline([
			("get_string", FunctionTransformer(extract_string_text, validate=False)),
			("token_tfidf_vec", TfidfVectorizer(analyzer="word", ngram_range=tok_ngram_range, 
									lowercase=True, min_df=0.1, norm='l2', max_features=2000)),
		])),
		("pos_tfidf", Pipeline([
			("get_tags", FunctionTransformer(extract_pos_tag_text, validate=False)),
			("pos_tfidf_vec", TfidfVectorizer(analyzer="word", ngram_range=pos_ngram_range, 
									lowercase=True, min_df=0.1, norm='l2', max_features=2000)),
		])),
		("stylo", Pipeline([
			("extract", StylometricFeatures(work_dir)), 
			("scale", StandardScaler())
		]))
	])

	for encoding in encodings:
		for level in levels:
			tic = time.perf_counter() 
			### Extract training features
			print(f'Starting {encoding} {level} {feat_combos} training features...')
			pipeline_file = os.path.join(out_dir, f'{encoding}_{level}_{feat_combos}_pipeline.pkl')
			X_train_file = os.path.join(out_dir, f'{encoding}_{level}_{feat_combos}_X_train.npz')
			y_train_file = os.path.join(out_dir, f'{encoding}_{level}_{feat_combos}_y_train.pkl')

			### Load or compute+save the doc_feature_pipeline
			if os.path.exists(pipeline_file):
				print("Loading saved pipeline...")
				doc_feature_pipeline = joblib.load(pipeline_file)
				pipeline_fitted = True
			else:
				print("No saved pipeline found. Will fit and save it when building training features.")
				pipeline_fitted = False
			
			### Load or compute+save training features
			if os.path.exists(X_train_file) and os.path.exists(y_train_file) and pipeline_fitted:
				print("Loading saved training features...")
				X_train = load_npz(X_train_file)
				y_train = joblib.load(y_train_file)
			else:
				print("Computing training features and fitting pipeline...") 
				train_trials_file = os.path.join(trials_dir, f'{encoding}_train_{level}_trials.npy')
				with open(train_trials_file, 'rb') as f:
					train_trials = np.load(f, allow_pickle=True) 
				train_pairs, y_train, train_texts = get_pairs_labels(train_trials) 
				train_doc_ids = sorted(set(d for pair in train_pairs for d in pair))
				doc2features_train, _ = build_doc_feature_matrix(train_doc_ids, train_texts, 
														doc_feature_pipeline, fit=True)
				X_train = build_pairwise_matrix(train_pairs, doc2features_train, feat_combos)
				save_npz(X_train_file, X_train)
				joblib.dump(y_train, y_train_file)
				joblib.dump(doc_feature_pipeline, pipeline_file)

			## Load or compute+save test features
			print(f'Starting test features...')
			X_test_file = os.path.join(out_dir, f'{encoding}_{level}_{feat_combos}_X_test.npz')
			y_test_file = os.path.join(out_dir, f'{encoding}_{level}_{feat_combos}_y_test.pkl')
			if os.path.exists(X_test_file) and os.path.exists(y_test_file) and pipeline_fitted:
				print("Loading saved test features...")
				X_test = load_npz(X_test_file)
				y_test = joblib.load(y_test_file)
			else:
				print("Computing test features...")
				test_trials_file = os.path.join(trials_dir, f'{encoding}_test_{level}_trials.npy')
				with open(test_trials_file, 'rb') as f:
					test_trials = np.load(f, allow_pickle=True) 
				test_pairs, y_test, test_texts = get_pairs_labels(test_trials)
				test_doc_ids = sorted(set(d for pair in test_pairs for d in pair))
				doc2features_test, _ = build_doc_feature_matrix(test_doc_ids, test_texts, 
													doc_feature_pipeline, fit=False)
				X_test = build_pairwise_matrix(test_pairs, doc2features_test, feat_combos)
				save_npz(X_test_file, X_test)
				joblib.dump(y_test, y_test_file)
			
			### Get feature names 
			all_feat_names = get_feat_names(doc_feature_pipeline, feat_combos)

			### Fit or load classifier
			clf_outfile = os.path.join(out_dir, 
					f'{encoding}_{level}_{feat_combos}_clf_{which_clf}.joblib')
			if os.path.exists(clf_outfile):
				print("Loading classifier...")
				clf = joblib.load(clf_outfile)
			else:
				print("Fitting classifier...")
				### Train classifier
				print(f'Running logistic regression classifier...')
				clf = LogisticRegression(max_iter=1000, random_state=19) 
				clf.fit(X_train, y_train)
				dump(clf, clf_outfile)

			### Evaluate + compute metrics
			acc, roc_auc, eer = compute_metrics(clf, X_test, y_test)

			### Output results
			output_file = os.path.join(out_dir, 
					f'{encoding}_{level}_{feat_combos}_{which_clf}_results.txt') 
			with open(output_file, 'w') as f:
				f.write(f"Encoding: {encoding}\n")
				f.write(f"Level: {level}\n")
				f.write(f"Feature combination: {feat_combos}\n")
				f.write(f"Classifier: {which_clf}\n")
				f.write(f"Number of training pairs: {X_train.shape[0]}\n")
				f.write(f"Number of test pairs: {X_test.shape[0]}\n")
				f.write(f"Number of features: {len(all_feat_names)}\n")
				f.write("----------------------------------------------\n")
				f.write(f"Accuracy: {acc}\n")
				f.write(f"ROC AUC: {roc_auc}\n")
				f.write(f"EER: {eer}\n")
			
			importances = clf.coef_[0]
			if len(importances) != len(all_feat_names): 
				print("Warning: Feature names and importances length mismatch")
				top_indices = np.argsort(np.abs(importances))[::-1][:10]
				for i in top_indices:
					print(f"Feature {i}: importance={importances[i]:.4f}")
			else:
				impt_outfile = os.path.join(out_dir, 
					f'{encoding}_{level}_{feat_combos}_{which_clf}_feat_impts.csv')
				importance_df = pd.DataFrame({
					"feature": all_feat_names,
					"importance": importances
					})
				importance_df = importance_df.sort_values('importance', ascending=False)
				importance_df.to_csv(impt_outfile, index=False)
			
			toc = time.perf_counter()
			print(f"Ran {encoding} {level} {feat_combos} {which_clf} stylometric pipeline in {(toc - tic)/60:0.3f} minutes")
	return


if __name__ == '__main__':
	try:
		yaml_path = sys.argv[1]
	except:
		print(f"Usage: {sys.argv[0]} [CONFIG_PATH]")

	cfg = yaml.safe_load(open(yaml_path)) 
	main(cfg)


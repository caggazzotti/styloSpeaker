"""
Extract top features from multiple CSV files containing feature importance scores per setting and create heatmaps and plots for easier comparison across settings
"""

import os
import sys
import yaml
import pandas as pd
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt


####----------------------------------Read in top features------------------------------####

### Read each CSV and extract the top features by setting and unique features
def get_top_feats(csv_files, level, feat_combos, plot_out_dir):
	## Dictionary to hold top features per setting
	top_feats_by_setting = {}

	for file in csv_files:
		setting_name = os.path.splitext(os.path.basename(file))[0]
		clean = setting_name.split('_')
		setting_clean = ' '.join([clean[0].upper(),clean[1].capitalize()]) #pretty name

		df = pd.read_csv(file)
		if feat_combos == 'diffabs': #clean up names since they all have the same prefix
			# Remove prefix up to first underscore
			df['feature'] = df['feature'].apply(lambda x: x.split('_', 1)[1] if '_' in x else x)

		## Sort by absolute importance
		df["abs_impt"] = df["importance"].abs()
		df = df.sort_values("abs_impt", ascending=False)

		top_20 = df.head(20)
		top_feats_by_setting[setting_clean] = top_20['feature'].tolist()

	## Convert to DataFrame: each column = a setting, each row = top-k feature
	comparison_df = pd.DataFrame(top_feats_by_setting)

	## Optional: save to CSV
	outfile = os.path.join(plot_out_dir, f'feat_comparison_{level}_{feat_combos}_top_20.csv')
	comparison_df.to_csv(outfile, index=False)
	print(comparison_df.head())

	## Make a long list of all unique top 20 features across all settings
	all_unique_features = sorted(set(sum(top_feats_by_setting.values(), [])))
	return all_unique_features, top_feats_by_setting


####----------------------------------Create plots------------------------------####

####----------- RANK MATRIX
def get_rank_heatmap(all_unique_features, top_feats_by_setting, level, feat_combos, heat_out_dir):
	## Initialize a DataFrame for ranks, default to NaN
	rank_matrix = pd.DataFrame(
		float('nan'), index=all_unique_features, columns=top_feats_by_setting.keys()
	)

	## Fill in ranks
	for setting, features in top_feats_by_setting.items():
		for rank, feature in enumerate(features):
			rank_matrix.loc[feature, setting] = rank + 1  #rank 1 to 20

	sort_col = rank_matrix.columns[0] #sort by BBN (col 1)
	rank_matrix[sort_col] = pd.to_numeric(rank_matrix[sort_col], errors='coerce') #ensure numeric dtype

	## Sort by BBN rank, NaNs go to the bottom
	rank_matrix_sorted = rank_matrix.sort_values(by=sort_col, na_position='last')

	## Plot heatmap with reversed colormap (lower rank = darker)
	plt.figure(figsize=(12, len(rank_matrix_sorted) * 0.3)) 
	sns.heatmap(rank_matrix_sorted, cmap='YlGnBu_r', linewidths=0.5, linecolor='gray', 
			 annot=True, fmt=".0f", cbar=False) 
	
	## If want a colorbar, flip it so rank 1 is at the top
	# ax = sns.heatmap(rank_matrix_sorted, cmap='YlGnBu_r', linewidths=0.5, linecolor='gray', 
	# 		 annot=True, fmt=".0f")
	# cbar = ax.collections[0].colorbar
	# cbar.ax.invert_yaxis()

	plt.title("Absolute Feature Rank (1=Most Important) for Each Setting")
	plt.xlabel("Setting")
	plt.ylabel("Feature")
	plt.tight_layout()
	rank_outfile = os.path.join(heat_out_dir, f'heatmap_feature_rank_{level}_{feat_combos}.png')
	plt.savefig(rank_outfile, dpi=300)
	return 


####----------- TOP FEATURES FOR NEGATIVE VS POSITIVE TRIALS
def get_top_posneg_plots(file, out_dir, level, feat_combos):
	setting_name = os.path.splitext(os.path.basename(file))[0]
	clean = setting_name.split('_')
	setting_clean = ' '.join([clean[0].upper(),clean[1].capitalize()]) #pretty name

	importance_df = pd.read_csv(file)
	if feat_combos == 'diffabs': #clean up names since they all have the same prefix
		## Remove prefix up to first underscore
		importance_df['feature'] = importance_df['feature'].apply(lambda x: x.split('_', 1)[1] if '_' in x else x)

	importance_df["abs_impt"] = importance_df["importance"].abs()
	importance_df = importance_df.sort_values("abs_impt", ascending=False)

	top_k = 15 #number of features to show

	## Top positive and negative features, sorted by absolute coef
	top_positive = importance_df[importance_df["importance"] > 0].nlargest(top_k, "abs_impt")
	top_negative = importance_df[importance_df["importance"] < 0].nlargest(top_k, "abs_impt")

	## Combine both into one plot-friendly DataFrame
	top_pos_neg = pd.concat([top_positive, top_negative])

	## Sort combined DataFrame by coefficient values (descending)
	top_pos_neg = top_pos_neg.sort_values(by="abs_impt", ascending=True) #ascending so most important at top

	## Plot
	plt.figure(figsize=(10, 8))
	plt.barh(top_pos_neg['feature'], top_pos_neg['importance'])
	plt.axvline(0, color='black', linewidth=0.8)  #separator line at zero
	plt.title(f'Top Features for Negative vs. Positive Trials (Not Absolute): {setting_clean}')
	plt.xlabel('Coefficient Value')
	plt.ylabel('Feature')
	plt.tight_layout()
	plt.show()
	plot_outfile = os.path.join(out_dir, 'feature_analysis', 
						f'{clean[0]}_{level}_{feat_combos}_posneg_plot.png')
	plt.savefig(plot_outfile)
	return 


####----------------------------------------MAIN---------------------------------------------####

def main(cfg):
	### Get config parameters
	out_dir = cfg['out_dir']
	encodings = cfg['encodings'] 
	levels = cfg['levels']
	feat_combos = cfg['feat_combos']

	feat_combos_name = f'_{feat_combos}_' #need _ to distinguish diffabs from concatsdiffabs and diffabsprod

	plot_out_dir = os.path.join(out_dir, 'feature_analysis')
	if not os.path.exists(plot_out_dir):
		os.makedirs(plot_out_dir)

	## Define the path to the folder containing CSV files
	csv_files_all = glob(os.path.join(out_dir, '*.csv'))
	csv_files = [filename for filename in csv_files_all if 'feat_impts' in filename and feat_combos_name in filename]

	for level in levels:
		level_files = []
		if level == 'hard':
			l = 'hard_' #to avoid matching 'harder'
		else:
			l = level
		for encoding in encodings:
			for file in csv_files:
				if l in file and encoding in file:
					level_files.append(file)

		### For heatmaps
		all_unique_features, top_feats_by_setting = get_top_feats(level_files, 
													level, feat_combos, plot_out_dir)
		get_rank_heatmap(all_unique_features, top_feats_by_setting, level, feat_combos, 
				   plot_out_dir)

		### For top positive/negative features per setting
		for file in level_files:
			get_top_posneg_plots(file, out_dir, level, feat_combos)
	return


if __name__ == '__main__':
	try:
		yaml_path = sys.argv[1]
	except:
		print(f"Usage: {sys.argv[0]} [CONFIG_PATH]")

	cfg = yaml.safe_load(open(yaml_path)) 
	main(cfg)
#!/bin/env python

#VERSION = "0.0.1"
#BUILD   = 4

#--------------------------------
# GRAPE: Genetic interaction Regression Analysis of Pairwise Effects
#
# MIT License
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
#---------------------------------


# ------------------------------------
# python modules
# ------------------------------------
import sys as sys
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multitest import fdrcorrection
from sklearn.linear_model import LinearRegression


def load_readcount_matrix(filepath, index_column=0, delimiter='\t'):
    """
    Load a user-generated fitness matrix instead of generating a synthetic one
    :param filepath: The path to the file to be loaded
    :param index_column: The column to use as an index (contains target IDs). Default 0.
    :param delimiter: tab or comma delimited file. Default '\t'
    :return: reads_df: A dataframe containing the readcount matrix. Index is unique ID, first column is target.
    """  
    reads_df = pd.read_csv(filepath, index_col=index_column, delimiter='\t')
    return reads_df

def get_foldchange_matrix(reads_df, control_columns, min_reads=0, pseudocount=1):
	"""
    Given a dataframe of raw read counts,
    1. filter for T0 min read counts.
    2. calculate log2 ratio of each sample relative to the mean of control columns.
    
    Parameters:
    - reads_df: dataframe of read counts. Presumably the index is a unique id and there is a target-gene column.
    - control_columns: Either a comma-separated list of control columns indices or a list of control sample labels.
    - min_reads: Minimum read count threshold for filtering samples. Not currently implemented. (How do we handle
      this across multiple control columns?) Default=0.
	-  pseudocount: Pseudocount added to the read counts to prevent division by or log of zero.
    """
    # control_columns to actual column names

	try:
		column_list = list(map(int, control_columns))
		control_column_labels = reads.columns.values[column_list]
	except ValueError:
		control_column_labels = control_columns

	print("Using controls: " + ",".join(map(str, control_column_labels)))

	target_column_labels = [x for x in reads_df.columns.values if x not in control_column_labels ]
	# target_column_labels[0] should still be the target gene

	ctrl_sum = reads_df[ control_column_labels ].sum(axis=1)
	fc_df = pd.DataFrame( index=reads_df.index.values, columns=target_column_labels, data=0.)
	fc_df[ target_column_labels[0] ] = reads_df[ target_column_labels[0] ] # we hope ths is the target column
	for col_name in target_column_labels[1:]:
		fc_df[col_name] = np.log2( ( (reads_df[col_name].values + pseudocount)/sum( reads_df[col_name].values ) ) /
		                           ( ( ctrl_sum + pseudocount ) / sum(ctrl_sum) ) )

	return fc_df

def get_mean_foldchange(fc_df, target_columns=0, mean_replicates=True, groupby_targets=True):
	"""
    Given a dataframe of fold changes, return the mean across replicates and/or the mean
    across guides targeting the same gene(s)
    
    Parameters:
    - fc_df: dataframe of fold changes, where index is unique ID and first column is the target gene(s). Subsequent
      columns are the replicates to be averaged.
    - target_columns: List of column labels indicating replicates to be averaged.
    - mean_replicates: whether to average across replicate columns. Default=True
    - groupby_targets: whether to groupby(target gene).mean(). Default=True
    """
	if target_columns:
		#
		# use target columns
		target_columns = target_columns
	else:
		# use all columns
		target_columns = fc_df.columns.values

	if mean_replicates:
		outcols = [fc_df.columns.values[0], 'meanFC']
	else:
		outcols = fc_df.columns.values

	mean_fc_df = pd.DataFrame( index = fc_df.index.values, columns=outcols, data=0.)
	mean_fc_df[ outcols[0] ] = fc_df[ outcols[0] ]

	if mean_replicates:
		mean_fc_df[ outcols[1] ] = fc_df[ target_columns ].mean(1)
	else:
		mean_fc_df[ outcols[1:] ] = fc_df[ fc_df.columns.values[1:] ]

	if groupby_targets:
		mean_fc_df = mean_fc_df.groupby( outcols[0] ).mean()

	return mean_fc_df

def load_genelist(filepath, sep='\t', index_col=0, names=['Genes']):
	gene_df = pd.read_table(filepath, sep=sep, index_col=index_col, names=names)
	#
	# can be used for reference genes or regression gene pairs
	#
	return gene_df

def mode_center(mean_fc_df):
	# assumes a polished fold change df where the index is the target gene(s). 
	# set mode to zero across entire distribution
	xx = np.linspace(-5, 4, 901)
	kx = stats.gaussian_kde(mean_fc_df[ mean_fc_df.columns[0] ])
	mode_x = xx[ np.argmax(kx.evaluate(xx)) ]
	return mean_fc_df - mode_x

def mode_center_vs_reference_genes(mean_fc_df, noness_genes):
	# assumes a polished fold change df where the index is the target gene(s),
	# NOT the unique guide ID
	# (not currently implemented)
	nonidx = [x for x in mean_fc_df.index.values if x in noness_genes]
	modecenter_fc_df = mean_fc_df - mean_fc_df.loc[nonidx].median()
	return modecenter_fc_df

def make_predictor_matrix( fc_df, target_gene_list, gene_delim='_'):
	predictor_matrix = pd.DataFrame( index=fc_df.index.values, columns=target_gene_list, data=0.)
	for target_array in fc_df.index.values:
		if gene_delim in target_array:
			#
			# this target is a gene pair
			#
			target_genes_in_array = target_array.split(gene_delim)
			if ( len( np.intersect1d(target_genes_in_array, target_gene_list) ) == len(target_genes_in_array) ):
				# both/all genes in the target array are in the genetic interaction gene list. include this genepair.
				predictor_matrix.loc[target_array, target_genes_in_array] = 1
		else:
			# absence of delimiter implies single gene knockout. is the single gene in our target_gene_list?
			if (target_array in target_gene_list):
				predictor_matrix.loc[target_array, target_array] = 1
	
	# Drop rows not targeting a gene in the target list
	dropme = np.where( predictor_matrix.sum(1)==0 )[0]  
	predictor_matrix.drop( predictor_matrix.index.values[dropme], axis=0, inplace=True )

	# Drop columns (genes) with no arrays targeting that gene
	dropme = np.where( predictor_matrix.sum(0)==0 )[0]  
	predictor_matrix.drop( predictor_matrix.columns.values[dropme], axis=1, inplace=True )

	obs_vector = fc_df.loc[ predictor_matrix.index.values ]
	print( 'regression matrix rows: {:5d}, cols: {:3d}'.format(predictor_matrix.shape[0], predictor_matrix.shape[1]) )
	return predictor_matrix, obs_vector

def filter_predictor():
	# dummy function. should this be a function?
	return

def do_regression( predictor_matrix, obs_vector, fit_intercept=False, delimiter='_' ):
	"""
    Calculate the regression and provide the initial GI score.
    Returns prediction (df), model (Logistic Regression object)

    Parameters:
    - predictor_matrix: dataframe of predictors. Usually binary. Columns = single genes, rows=single and double genes.
    - obs_vector: dataframe of observed fold change, index is the same as predictor_matrix
    - fit_intercept: whether to fit the intercept. Default false.
    """
	model = LinearRegression( fit_intercept=fit_intercept ).fit( predictor_matrix.values, obs_vector.values )
	pred_fc = model.predict( predictor_matrix.values )
	pairs = pd.DataFrame( index=predictor_matrix.index.values, columns=['fc_obs','fc_exp','GI_raw','g1_fc','g2_fc','dLFC'], data=0.)
	pairs['fc_obs']  = obs_vector.values
	pairs['fc_exp'] = pred_fc.flatten()  # already an array
	# remove singles: columns of predictor matrix
	single_genes = predictor_matrix.columns.values

	singles = pairs.loc[single_genes,['fc_obs','fc_exp']]
	pairs.drop( single_genes, axis=0, inplace=True )

	pairs['GI_raw'] = pairs.fc_obs - pairs.fc_exp
	#
	# now for each pair, get g1, g2, dLFC based on observed FC in singles
	for genepair in pairs.index.values:
		g1, g2 = genepair.split(delimiter)
		g1_fc = singles.loc[g1,'fc_obs']
		g2_fc = singles.loc[g2,'fc_obs']
		dLFC = pairs.loc[genepair, 'fc_obs'] - (g1_fc + g2_fc)
		pairs.loc[genepair, ['g1_fc','g2_fc','dLFC']] = g1_fc, g2_fc, dLFC
	#
	# other data:
	#
	metadata = {}
	metadata['Rsq'] = model.score( predictor_matrix.values, obs_vector.values)
	metadata['Intercept'] = model.intercept_
	metadata['Params']  = model.get_params()
	return pairs, singles, metadata

def dynamic_range_filter( regression_pairs ):
	"""
	Identify target list where expected phenotype (fc_exp) is beyond the dynamic range of the assay (min fc_obs).
	These break the regression model.

	Parameters:
	- regression_pairs: 'pairs' output from do_regression().
	Returns:
	- subset of regression_pairs dataframe whose indices should be deleted from the foldchange df, and the 
	  grape pipeline run again.
	"""
	fc_limit = regression_pairs['fc_obs'].min()
	pairs_to_remove = regression_pairs[ regression_pairs['fc_exp'] < fc_limit ]
	return pairs_to_remove

def get_zscore( regression_df, half_window_size=500, monotone_filter=False ):
	"""
	calculate zscore of genetic interactions through drugz variance window method

	Parameters: 
	- pairs: output of do_regression()
	- half_window_size: half window size for calculating local variance. If set to 
						zero (default), calculate global, not local variance
	- monotonte filter: force monotonic increase in variance
	Returns:
	- pairs, with 'local_std', 'GI_Zscore', 'Pval_synth', 'Padj_synth', 'Pval_supp', 'Padj_supp' columns
	"""

	# sort the regression DF by expected fold change:

	zscore_df = regression_df.sort_values('fc_exp', ascending=False).copy()

	# Intialize output columns
	zscore_df[['local_std','GI_Zscore']] = 0.

	if (half_window_size==0):
		#
		# if half_window_size = 0, use global instead of local Z score
		#
		zscore_df['local_std'] = zscore_df['GI_raw'].std()
		zscore_df['GI_Zscore'] = stats.zscore( zscore_df.GI_raw.values )

	else:
		#
		# otherwise step through the data and calculate local std
		#
		stepsize = int( np.ceil( half_window_size / 5) )
		for idx in range( half_window_size, len(zscore_df) - half_window_size, stepsize):
			#
			# select the window slice
			#
			bin_ = zscore_df.iloc[idx - half_window_size : idx + half_window_size ].copy()
			#
			# remove outliers and calculate std
			#
			Q1,Q3 = bin_['GI_raw'].quantile([0.25,0.75])
			IQR = Q3 - Q1
			lower_bound = Q1 - 1.5*IQR
			upper_bound = Q3 + 1.5*IQR
			bin_ = bin_[(bin_['GI_raw'] >= lower_bound) & (bin_['GI_raw'] <= upper_bound)]
			local_std = bin_['GI_raw'].std()
			#
			# assigne value in df, 
			#
			if (monotone_filter):
				prev_std = zscore_df.iloc[idx-1, zscore_df.columns.get_loc('local_std')]
				zscore_df.iloc[idx:idx+stepsize, zscore_df.columns.get_loc('local_std')]=max(local_std, prev_std)
			else:
				zscore_df.iloc[idx:idx+stepsize, zscore_df.columns.get_loc('local_std')]=local_std
		#
		# now for the first (0..half_window_size) and last segments of the df
		#
		zscore_df.iloc[:half_window_size, zscore_df.columns.get_loc('local_std')] = zscore_df.iloc[half_window_size, zscore_df.columns.get_loc('local_std')]
		zscore_df.iloc[-half_window_size:, zscore_df.columns.get_loc('local_std')]= zscore_df.iloc[-half_window_size-1, zscore_df.columns.get_loc('local_std')]
		#
		# and calculate z score
		#
		zscore_df['GI_Zscore'] = zscore_df['GI_raw'] / zscore_df['local_std']   # mean had better be zero!
	#
	# calculate synthetic and suppressing interaction p-value and FDR
	#
	zscore_df.sort_values('GI_Zscore', ascending=True, inplace=True)
	zscore_df['Pval_synth'] = stats.norm.cdf( zscore_df.GI_Zscore )
	zscore_df['Padj_synth'] = fdrcorrection( zscore_df.Pval_synth )[1] 
    
	zscore_df.sort_values('GI_Zscore', ascending=False, inplace=True)
	zscore_df['Pval_supp'] = stats.norm.sf( zscore_df.GI_Zscore )
	zscore_df['Padj_supp'] = fdrcorrection( zscore_df.Pval_supp )[1]
    
	return zscore_df.sort_values('GI_Zscore', ascending=True)


def get_args():
    # get some user arguments
    return 0


def main():
    args = get_args()
    grape(args)


if __name__ == "__main__":
	# don't do this yet.
    main()

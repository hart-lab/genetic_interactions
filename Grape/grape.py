#!/bin/env python

#VERSION = "0.0.1"
#BUILD   = 2

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
    - target_columns: Column labels or indices indicating replicates to be averaged. Not currently implemented.
    - mean_replicates: whether to average across replicate columns. Default=True
    - groupby_targets: whether to groupby(target gene).mean(). Default=True
    """
	if mean_replicates:
		outcols = [fc_df.columns.values[0], 'meanFC']
	else:
		outcols = fc_df.columns.values

	mean_fc_df = pd.DataFrame( index = fc_df.index.values, columns=outcols, data=0.)
	mean_fc_df[ outcols[0] ] = fc_df[ outcols[0] ]

	if mean_replicates:
		mean_fc_df[ outcols[1] ] = fc_df[ fc_df.columns.values[1:] ].mean(1)
	else:
		mean_fc_df[ outcols[1:] ] = fc_df[ fc_df.columns.values[1:] ]

	if groupby_targets:
		mean_fc_df = mean_fc_df.groupby( outcols[0] ).mean()

	return mean_fc_df

def load_genelist(filepath, sep='\t', index_col=0):
	gene_df = pd.read_table(filepath, sep=sep, index_col=index_col)
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
	nonidx = [x for x in mean_fc_df.index.values if x in noness_genes]
	modecenter_fc_df = mean_fc_df - mean_fc_df.loc[nonidx].median()
	return modecenter_fc_df


def get_args():
    # get some user arguments
    return 0


def main():
    args = get_args()
    grape(args)


if __name__ == "__main__":
	# don't do this yet.
    main()

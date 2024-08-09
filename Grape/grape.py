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

def get_foldchange_matrix(reads_df, control_columns, min_reads=0, pseudocount=0):
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




def get_args():
    # get some user arguments
    return 0


def main():
    args = get_args()
    grape(args)


if __name__ == "__main__":
	# don't do this yet.
    main()

#!/bin/env python

#VERSION = "0.0.2"
#BUILD   = 2


#--------------------------------
# SYNULATOR: synthetic data for synthetic lethal CRISPR screens
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

def load_fitness_matrix(filepath, index_column=0, delimiter='\t'):
    """
    Load a user-generated fitness matrix instead of generating a synthetic one
    :param filepath: The path to the file to be loaded
    :param index_column: The column to use as an index (contains target IDs). Default 0.
    :param delimiter: tab or comma delimited file. Default '\t'
    :return: fit_mat: A dataframe containing the N x N fitness matrix to be used in the further analysis
    """  
    fitness_matrix = pd.read_csv(filepath, index_col=index_column, delimiter='\t')
    return fitness_matrix

def generate_fitness_matrix(num_total_genes=200,
                            num_fitness_genes=50,
                            mu_k_wt=1.0,
                            mu_k_fitness_min=0.5,
                            mu_k_fitness_max=1.0,
                            genetic_interaction_frequency=0.03,
                            genetic_interaction_fitness_min=0.5,
                            genetic_interaction_fitness_max=1.0):
    """
    Generate an NxN matrix of target knockout fitness values.
    A[i,i] = single gene fitness, where 1.0 = wildtype and 0.0 = total loss of proliferation
    A[i,j] = genetic interaction fitness, where 1.0 = no interaction and 0.0 = synthetic lethal.
    In the master fitness equation Xc = Xo*2^(kt), where k~N(mu,sigma), the mu term is generated
    from these fitness values
    :param filepath: The path to the file to be loaded
    :param index_column: The column to use as an index (contains target IDs). Default 0.
    :param delimiter: tab or comma delimited file. Default '\t'
    :return: fit_mat: A dataframe containing the N x N fitness matrix to be used in the further analysis
    """  
    # initialize matrix: all ones
    fitness_matrix = pd.DataFrame( index=np.arange(num_total_genes), columns=np.arange(num_total_genes),
                                    data=np.ones([num_total_genes,num_total_genes]) )
    num_wt_genes = num_total_genes - num_fitness_genes
    if (num_fitness_genes > num_total_genes):
        sys.exit('Number of fitness genes exceeds number of total genes\n')
    #
    # single gene phenotypes: mu_k. 
    # wildtype genes (no fitness phenotype)
    #
    for i in np.arange(num_wt_genes):
        fitness_matrix.loc[i,i] = mu_k_wt
    #
    # fitness genes (some ko fitness phenotype)
    # mu_k is uniformly distributed on (mu_k_fitness_min, mu_k_fitness_max)
    #
    for i in np.arange(num_wt_genes,num_total_genes):
        fitness_matrix.loc[i,i] = np.random.uniform(low=mu_k_fitness_min, high=mu_k_fitness_max)   
    #
    # off diagonal: genetic interaction. 
    #
    for i in range(num_total_genes-1):
        for j in range(i+1,num_total_genes):
            if (np.random.rand() <= genetic_interaction_frequency):
                fitness_matrix.loc[i,j] = np.random.uniform(low=genetic_interaction_fitness_min, high=genetic_interaction_fitness_max)
    #
    # return fitness matrix
    #
    return fitness_matrix

def generate_fitness_table(fitness_matrix,
                            sigma_k=0.05,
                            t=8,
                            transduction_depth=500,
                            median_read_depth=500,
                            pseudocount=0,
                            set_seed=0,
                            seed=0):
    """
    Generate a dataframe of targets, single and double, with expected and actual knockout fitness values.
    :param fitness_matrix: matrix from generate_fitness_matrix()
    :return: fitness_table: A dataframe containing the total screen simulation
    """
    if (set_seed==1):
       np.random.seed(seed)

    genelist = fitness_matrix.index.values
    num_genes_in_genelist = len(genelist)
    num_total_genepairs_and_singles = (num_genes_in_genelist * (num_genes_in_genelist -1 ) / 2) + num_genes_in_genelist
    fitness_table = pd.DataFrame( index=np.arange(num_total_genepairs_and_singles), columns=['target_id','X0','mu_k','GI'], data=0.)
                                                        # set data=0. to set column dtypes to float
    fitness_table = fitness_table.astype({'target_id': object}, copy=False, errors='raise')     # if the genelist is integers,
                                                                                                # we should interpret as strings.
    #
    # populate the fitness table
    #
    idx = 0
    for i in np.arange(num_genes_in_genelist):
        gene1 = str(genelist[i])
        for j in np.arange(i,num_genes_in_genelist):
            gene2 = str(genelist[j])
            if (i==j):
                #
                # on the diagonal of the fitness matrix, we are dealing with single knockout fitness
                #
                mu_k = fitness_matrix.loc[i,i]
                target = gene1
                gi = 1.
            else:
                #
                # off the diagonal, we are dealing with double knockout fitness. Under the multiplicatiave
                # model, this is ki * kj. With interactions, this is ki * kj * GI(i,j)
                #
                mu_k = fitness_matrix.loc[i,i] * fitness_matrix.loc[j,j] * fitness_matrix.loc[i,j]   # k1 * k2 * GI
                target = gene1 + "_" + gene2
                gi = fitness_matrix.loc[i,j]
            fitness_table.loc[idx,['target_id','mu_k','GI']] = target, mu_k, gi
            idx = idx + 1                 # pretty sure there's a better way to do this.
    #
    # set X0 counts for each target. Placeholder for future roadmap where this value might vary.
    #
    fitness_table['X0'] = transduction_depth + pseudocount
    #
    # we have mu_k for each target, and sigma_k as a noise term. 
    # Calculate Xt = X0 * 2^(kt), where k~N(mu_k, sigma_k)
    
    fitness_table['k_obs'] = np.random.normal( loc=fitness_table['mu_k'].values, scale=sigma_k)
    
    fitness_table['Xt']    = fitness_table.X0 * 2**( fitness_table.k_obs * t )
    
    # scale observed cell counts, Xt, to median library coverage/sequecing read depth and add pseudocount, if present
    
    fitness_table['Reads_t']   = np.floor( fitness_table.Xt.values * median_read_depth / fitness_table.Xt.median() ) + pseudocount
    fitness_table['Log2fc']   =  np.log2(  (fitness_table.Reads_t.values / sum(fitness_table.Reads_t.values ) ) /  \
                                        (fitness_table.X0.values / sum(fitness_table.X0.values ) ) )
    
    fitness_table = fitness_table.astype({'Reads_t': int}, copy=False, errors='raise')     # reads should be integer
    fitness_table.set_index('target_id', inplace=True)

    # complete?

    return fitness_table

def write_fitness_table(fitness_table, output_filepath, float_format='%4.3f', sep='\t'):
    fitness_table.to_csv(output_filepath, float_format=float_format, sep=sep)
    return 0

def synulator(args):
    # lots of parsing of user arguments to be processed here
    fitness_matrix = generate_fitness_matrix()
    fitness_table  = generate_fitness_table(fitness_matrix=fitness_matrix)
    write_fitness_table(fitness_table, output_filepath='./simulated_gi_screen.txt')
    return 0

def get_args():
    # get some user arguments
    return 0

def main():
    args = get_args()
    synulator(args)

    
if __name__ == "__main__":
    main()
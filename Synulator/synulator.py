#!/bin/env python

#VERSION = "0.0.3"
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
import scipy.stats as stats
import argparse

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
                            mu_k_fitness_min=0.2,
                            mu_k_fitness_max=1.0,
                            genetic_interaction_frequency=0.01,
                            genetic_interaction_fitness_min=0.2,
                            genetic_interaction_fitness_max=1.0,
                            wt_gi_multiplier=0.1):
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
    # Initialize gene label
    wt_genes = list(range(num_wt_genes))
    fitness_genes = list(range(num_wt_genes, num_total_genes))
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
        fit_i = fitness_matrix.loc[i,i]
        for j in range(i+1,num_total_genes):
            fit_j = fitness_matrix.loc[j,j]
            if ( (fit_i ==1) & (fit_j ==1 ) ):
                # among "wildtype" genes, reduce 
                genetic_interaction_frequency_threshold = genetic_interaction_frequency * wt_gi_multiplier
            else:
                genetic_interaction_frequency_threshold = genetic_interaction_frequency
            if (np.random.rand() <= genetic_interaction_frequency_threshold):
                fitness_matrix.loc[i,j] = np.random.uniform(low=genetic_interaction_fitness_min, high=genetic_interaction_fitness_max)
    #
    # Return fitness matrix and gene labels
    #
    gene_labels = {
        'wildtype': wt_genes,
        'fitness': fitness_genes
    }

    return fitness_matrix, gene_labels


def generate_fitness_table( fitness_matrix,
                            gene_labels,
                            num_guides=4,
                            guide_stddev=0.0755,
                            sigma_k=0.03,
                            t=8,
                            transduction_depth=500,
                            median_read_depth=500,
                            overdispersion_param=0.5,
                            pseudocount=1,
                            set_seed=0,
                            seed=0):
    """
    Generate a guide-level fitness table.
    :parm fitness_matrix: matrix from generate_fitness_matrix()
    :parm gene_labels: dictionary containing 'wildtype' or 'fitness' gene lists, from generate_fitness_matrix().
    :parm num_guides: Number of guides per target.
    :parm std_dev: Standard deviation for sampling mu_k values.
    :parm overdispersion_param: must be between 0 and 1.
    
    :return: guide_level_fitness_table: DataFrame containing guide-level fitness information.
    """
    if (set_seed==1):
       np.random.seed(seed)

    genelist = fitness_matrix.index.values
    num_genes_in_genelist = len(genelist)
    guide_level_data = []

    #
    # populate the guide-level fitness table
    #
    for i in range(num_genes_in_genelist):
        gene1 = str(genelist[i])
        gene1_label = 'wildtype' if i in gene_labels['wildtype'] else 'fitness'
        for j in range(i, num_genes_in_genelist):
            gene2 = str(genelist[j])
            gene2_label = 'wildtype' if j in gene_labels['wildtype'] else 'fitness'
            if (i==j):
                #
                # on the diagonal of the fitness matrix, we are dealing with single knockout fitness
                #
                mu_k = fitness_matrix.loc[i,i]
                target = gene1
                gi = 1.
                guide_label = gene1_label
            else:
                #
                # off the diagonal, we are dealing with double knockout fitness. Under the multiplicatiave
                # model, this is ki * kj. With interactions, this is ki * kj * GI(i,j)
                #
                mu_k = fitness_matrix.loc[i,i] * fitness_matrix.loc[j,j] * fitness_matrix.loc[i,j]   # k1 * k2 * GI
                target = f'{gene1}_{gene2}'
                gi = fitness_matrix.loc[i,j]
                guide_label = f'{gene1_label}_{gene2_label}'
                
            # generate guide-level data for each target
            for guide_num in range(1, num_guides + 1):
                guide_id = f'{target}_guide{guide_num}'
                mu_k_guide = np.random.normal(loc=mu_k, scale=guide_stddev)   # sample mu_k for guides
                guide_level_data.append({
                    'guide_id': guide_id,
                    'target_id': target,
                    'mu_k': mu_k_guide,
                    'GI': gi,
                    'label': guide_label
                })

    # Convert the list of dictionaries to a DataFrame
    fitness_table = pd.DataFrame(guide_level_data)
    fitness_table = fitness_table.astype({'mu_k': float, 'GI': float})

    #
    # set X0 counts for each target. Placeholder for future roadmap where this value might vary.
    #
    fitness_table['X0'] = transduction_depth + pseudocount
    #
    # we have mu_k for each target, and sigma_k as a noise term. 
    # Calculate Xt = X0 * 2^(kt), where k~N(mu_k, sigma_k)
    
    fitness_table['k_obs'] = np.random.normal( loc=fitness_table['mu_k'].values, scale=sigma_k)
    
    fitness_table['Xt'] = fitness_table.X0 * 2**( fitness_table.k_obs * t )
    
    # scale observed cell counts, Xt, to median library coverage/sequecing read depth
    reads_t = np.floor( fitness_table.Xt.values * median_read_depth / fitness_table.Xt.median() )

    # add sequencing noise from negative binomial model with overdispersion paramenter p
    p = overdispersion_param
    n = reads_t * p / (1-p)
    if np.any(n <= 0):
        raise ValueError("Invalid 'n' values calculated for the negative binomial distribution.")

    # calculate "observed" reads and add pseudocount, if present. 
    
    fitness_table['Reads_t']   = stats.nbinom.rvs(n=n, p=p) + pseudocount

    # calculate fold change from observed reads_t and initial reads X0.
    # note that absence of a pseudocount can lead to zeros at reads_t and nans for log(fc)

    fitness_table['Log2fc']   =  np.log2(  (fitness_table.Reads_t.values / sum(fitness_table.Reads_t.values ) ) /  \
                                        (fitness_table.X0.values / sum(fitness_table.X0.values ) ) )
    
    fitness_table = fitness_table.astype({'Reads_t': int}, copy=False, errors='raise')     # reads should be integer
    fitness_table.set_index('guide_id', inplace=True)

    # complete?

    return fitness_table


def write_fitness_table(fitness_table, output_filepath, float_format='%4.3f', sep='\t'):
    # select only numeric columns
    numeric_columns = fitness_table.select_dtypes(include=['float', 'int']).columns
    
    formatted_fitness_table = fitness_table.copy()
    formatted_fitness_table[numeric_columns] = formatted_fitness_table[numeric_columns].apply(lambda col: col.map(lambda x: float_format % x))
    formatted_fitness_table.to_csv(output_filepath, sep=sep, index=True)
    return 0


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Simulate genetic interaction screen data.")
    parser.add_argument('--output', type=str, default='./simulated_gi_screen.txt',
                        help="Output file path for the simulated fitness table. Default: ./simulated_gi_screen.txt")
    parser.add_argument('--num_total_genes', type=int, default=200,
                        help="Total number of genes. Default: 200")
    parser.add_argument('--num_fitness_genes', type=int, default=50,
                        help="Number of fitness genes. Default: 50")
    parser.add_argument('--mu_k_wt', type=float, default=1.0,
                        help="Wildtype fitness mean. Default: 1.0")
    parser.add_argument('--mu_k_fitness_min', type=float, default=0.2,
                        help="Minimum fitness value for fitness genes. Default: 0.2")
    parser.add_argument('--mu_k_fitness_max', type=float, default=1.0,
                        help="Maximum fitness value for fitness genes. Default: 1.0")
    parser.add_argument('--genetic_interaction_frequency', type=float, default=0.01,
                        help="Frequency of genetic interactions. Default: 0.01")
    parser.add_argument('--genetic_interaction_fitness_min', type=float, default=0.2,
                        help="Minimum fitness value for genetic interactions. Default: 0.2")
    parser.add_argument('--genetic_interaction_fitness_max', type=float, default=1.0,
                        help="Maximum fitness value for genetic interactions. Default: 1.0")
    parser.add_argument('--wt_gi_multiplier', type=float, default=0.1,
                        help="Multiplier for wildtype genetic interactions. Default: 0.1")
    parser.add_argument('--num_guides', type=int, default=4,
                        help="Number of guides per gene or pair. Default: 4")
    parser.add_argument('--guide_stddev', type=float, default=0.0755,
                        help="Standard deviation for guide-level fitness values. Default: 0.0755")
    parser.add_argument('--sigma_k', type=float, default=0.03,
                        help="Standard deviation for observed fitness noise. Default: 0.03")
    parser.add_argument('--time', type=int, default=8,
                        help="Doubling time. Default: 8")
    parser.add_argument('--transduction_depth', type=int, default=500,
                        help="Initial transduction depth. Default: 500")
    parser.add_argument('--median_read_depth', type=int, default=500,
                        help="Median read depth for normalization. Default: 500")
    parser.add_argument('--overdispersion_param', type=float, default=0.5,
                        help="Overdispersion parameter for sequencing noise. Default: 0.5")
    parser.add_argument('--pseudocount', type=int, default=1,
                        help="Pseudocount added to avoid zeroes. Default: 1")
    parser.add_argument('--set_seed', type=int, default=0,
                        help="Whether to set a random seed. Default: 0 (no seed)")
    parser.add_argument('--seed', type=int, default=0,
                        help="Seed value for random number generation. Default: 0")
    return parser.parse_args()


def synulator(args):
    """
    Simulate the genetic interaction screen using provided arguments.
    """
    # Generate fitness matrix and labels
    fitness_matrix, gene_labels = generate_fitness_matrix(
        num_total_genes=args.num_total_genes,
        num_fitness_genes=args.num_fitness_genes,
        mu_k_wt=args.mu_k_wt,
        mu_k_fitness_min=args.mu_k_fitness_min,
        mu_k_fitness_max=args.mu_k_fitness_max,
        genetic_interaction_frequency=args.genetic_interaction_frequency,
        genetic_interaction_fitness_min=args.genetic_interaction_fitness_min,
        genetic_interaction_fitness_max=args.genetic_interaction_fitness_max,
        wt_gi_multiplier=args.wt_gi_multiplier
    )
    
    # Generate guide-level fitness table
    fitness_table = generate_fitness_table(
        fitness_matrix=fitness_matrix,
        gene_labels=gene_labels,
        num_guides=args.num_guides,
        guide_stddev=args.guide_stddev,
        sigma_k=args.sigma_k,
        t=args.time,
        transduction_depth=args.transduction_depth,
        median_read_depth=args.median_read_depth,
        overdispersion_param=args.overdispersion_param,
        pseudocount=args.pseudocount,
        set_seed=args.set_seed,
        seed=args.seed
    )
    
    # Write the table to a file
    write_fitness_table(fitness_table, output_filepath=args.output)
    print(f"Simulation complete. Output saved to: {args.output}")
    return 0


def main():
    args = get_args()
    synulator(args)

    
if __name__ == "__main__":
    main()
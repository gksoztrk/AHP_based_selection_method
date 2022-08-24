import numpy as np 
from random import randint
import random
import math 
from copy import deepcopy
import collections
import pandas as pd
from random import uniform
import sys
from itertools import islice, combinations
from itertools import chain
import datetime
from scipy.spatial.distance import cdist
from time import time
from sklearn.cluster import KMeans
from itertools import permutations 
import statistics
from collections import defaultdict 

#AHP Starts

comparision_list=[1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1, 2, 3, 4, 5, 6, 7, 8, 9] #defined by Saaty (1987).

def pairwise_criteria (no_criteria, comparision_list): #Creates the Pairwise Comparison Matrices (PCMs) randomly
        
    comp_matrix = np.identity((no_criteria))
    for i in range(no_criteria):
        for j in range(i+1, no_criteria):
            value = random.choice(comparision_list)
            comp_matrix[i, j]=value
            comp_matrix[j, i]=1/comp_matrix[i, j]
   
    return (comp_matrix)


def normalized_pairwised_matrix (pairwised_matrix_C, no_criteria): #normise the PCMs
    column_sum=[]
    for j in range(pairwised_matrix_C.shape[1]):
        criteria = pairwised_matrix_C[:, j]
        tot=np.sum(criteria)
        column_sum.append(tot)

    pair_comp_matr=deepcopy(pairwised_matrix_C)
    for row in range(no_criteria):
        pair_comp_matr[:,row]=pair_comp_matr[:,row]/column_sum[row]

    return pair_comp_matr


def overall_priority_w (pair_comp_matr): #overall weights
    
    row_sum=pair_comp_matr.sum(axis=1)
    overall_priority=row_sum/len(pair_comp_matr)
    
    return overall_priority




def eigen_vector (no_criteria, comparision_list, RI, pairwised_mat, weights): #check if the randonly created weights are consistent. 

    eigen_matrix = np.identity((no_criteria))
    np.fill_diagonal(eigen_matrix, weights)

    for diag in range(pairwised_mat.shape[1]):
        for cell in range(pairwised_mat.shape[0]):
            if eigen_matrix[cell][diag] != 0:
                continue
            else:
                eigen_matrix[cell][diag] = weights[diag]*pairwised_mat[cell][diag]

    row_sum=eigen_matrix.sum(axis=1)

    lambda_max=[]
    for row in range (eigen_matrix.shape[0]):
        lam=row_sum[row]/weights[row] 
        lambda_max.append(lam)

    lambda_mean=statistics.mean(lambda_max)

    const_inx=(lambda_mean-no_criteria)/(no_criteria - 1)

    consist_ratio=const_inx/RI[no_criteria] #this ratio should be less then 0.10. 

    return consist_ratio



'''
Below function creates PCMs for each objective considered in the multi objective optimization algorithm. Each PCM is in the size 
of the number of solutions obtained by the multi-objective optimization. 

This function includes the rescaling method proposed.
'''
def options_matrix (final_frontier_crietria, comparision_list, final_frontier_crietria_all): 

    differences_list=[]
    for sol in final_frontier_crietria:
        differences=[]
        for dif in final_frontier_crietria:
            differ = sol - dif
            differences.append(differ)
        differences_list.append(differences) 

    difference_matrix = np.hstack(differences_list)
    difference_matrix = np.reshape(difference_matrix, (len(final_frontier_crietria), len(final_frontier_crietria)))


    max_exist=np.max(final_frontier_crietria_all)
    min_exist=np.min(final_frontier_crietria_all)

    interval=(max_exist-min_exist)/((len(comparision_list)+1)/2)
    assignment=[]
    for ass in range(int((len(comparision_list)+1)/2)): #rescale between the differences
        assign=((ass)*interval)
        assignment.append(assign) 
    assignment.append(max_exist-min_exist)


    opt_matrix = np.identity((len(final_frontier_crietria)))
    for j in range(len(difference_matrix)):
        for i in range(j+1, len(difference_matrix)):
            
            if assignment[0] <= difference_matrix[j, i] < assignment[1]:
                opt_matrix[j, i]=comparision_list[8]
                opt_matrix[i, j]=1/opt_matrix[j, i]
            elif assignment[1] <= difference_matrix[j, i] < assignment[2]:
                opt_matrix[j, i]=comparision_list[7]
                opt_matrix[i, j]=1/opt_matrix[j, i]
            elif assignment[2] <= difference_matrix[j, i] < assignment[3]:
                opt_matrix[j, i]=comparision_list[6]
                opt_matrix[i, j]=1/opt_matrix[j, i]
            elif assignment[3] <= difference_matrix[j, i] < assignment[4]:
                opt_matrix[j, i]=comparision_list[5]
                opt_matrix[i, j]=1/opt_matrix[j, i]
            elif assignment[4] <= difference_matrix[j, i] < assignment[5]:
                opt_matrix[j, i]=comparision_list[4]
                opt_matrix[i, j]=1/opt_matrix[j, i]
            elif assignment[5] <= difference_matrix[j, i] < assignment[6]:
                opt_matrix[j, i]=comparision_list[3]
                opt_matrix[i, j]=1/opt_matrix[j, i]
            elif assignment[6] <= difference_matrix[j, i] < assignment[7]:
                opt_matrix[j, i]=comparision_list[2]
                opt_matrix[i, j]=1/opt_matrix[j, i]
            elif assignment[7] <= difference_matrix[j, i] < assignment[8]:
                opt_matrix[j, i]=comparision_list[1]
                opt_matrix[i, j]=1/opt_matrix[j, i]
            elif assignment[8] <= difference_matrix[j, i] < assignment[9]:
                opt_matrix[j, i]=comparision_list[0]
                opt_matrix[i, j]=1/opt_matrix[j, i]

            elif -assignment[1] <= difference_matrix[j, i] < assignment[0]:
                opt_matrix[j, i]=comparision_list[8]
                opt_matrix[i, j]=1/opt_matrix[j, i]
            elif -assignment[1] >= difference_matrix[j, i] > -assignment[2]:
                opt_matrix[j, i]=comparision_list[9]
                opt_matrix[i, j]=1/opt_matrix[j, i]
            elif -assignment[2] >= difference_matrix[j, i] > -assignment[3]:
                opt_matrix[j, i]=comparision_list[10]
                opt_matrix[i, j]=1/opt_matrix[j, i]
            elif -assignment[3] >= difference_matrix[j, i] > -assignment[4]:
                opt_matrix[j, i]=comparision_list[11]
                opt_matrix[i, j]=1/opt_matrix[j, i]
            elif -assignment[4] >= difference_matrix[j, i] > -assignment[5]:
                opt_matrix[j, i]=comparision_list[12]
                opt_matrix[i, j]=1/opt_matrix[j, i]
            elif -assignment[5] >= difference_matrix[j, i] > -assignment[6]:
                opt_matrix[j, i]=comparision_list[13]
                opt_matrix[i, j]=1/opt_matrix[j, i]
            elif -assignment[6] >= difference_matrix[j, i] > -assignment[7]:
                opt_matrix[j, i]=comparision_list[14]
                opt_matrix[i, j]=1/opt_matrix[j, i]
            elif -assignment[7] >= difference_matrix[j, i] > -assignment[8]:
                opt_matrix[j, i]=comparision_list[15]
                opt_matrix[i, j]=1/opt_matrix[j, i]
            elif -assignment[8] >= difference_matrix[j, i] > -assignment[9]:
                opt_matrix[j, i]=comparision_list[16]
                opt_matrix[i, j]=1/opt_matrix[j, i]


    return (opt_matrix)
    
 #after this step, the PCMs created for the objectives should be normised, and weighted. Finally, these weights and the preference weights should be used in the scores function below. 

def scores (weights_equ, weights_comp, weights_cont, weights): #obtains the scores
        
        sco_lits=[]
        sco_lits.append(weights_equ)
        sco_lits.append(weights_comp)
        sco_lits.append(weights_cont)

        scores_matrix = np.hstack(sco_lits)
        scores_matrix = np.reshape(scores_matrix, (len(weights), len(weights_comp)))

        scores=[]
        for cols in range(len(weights_comp)):
            scor= np.matmul(scores_matrix[:, cols], weights)
            scores.append(scor)
        
        return scores

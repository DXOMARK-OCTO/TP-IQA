# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 11:22:05 2021

@author: nchahine
"""

import asapgpu as asapGPU
import scipy.sparse as ssp
import torch
import numpy as np

SIGMA_JOD = 1.4826

def updateCompMat(idx, compMat):
    """
    Updates the comparison matrix with new comparison output. 

    Parameters
    ----------
    idx : tuple
        index with the form (a,b) where always compMat[a,b] += 1.

    Returns
    -------
    compMat.

    """ 
    if idx:
        compMat[idx[0], idx[1]] += 1
    
    return compMat 
   
def predictNext(compMat, useCuda=False, useBatch=True, **kwargs):
    return asapGPU.ASAP(compMat, mst_mode=useBatch, cuda=useCuda).tolist()    
    
def inferScores(compMat, useCuda=False):
    # Create a sparse matrix
    sparseCompMat = ssp.coo_matrix(compMat)

    # Check that the matrix is square
    rows, cols = sparseCompMat.shape
    
    if rows != cols:
        raise ValueError("The comparison matrix must be square.")
    
    # convert the matrix to tensor to have G(rows: condition1, condition2, comparison_outcomes, cols: pairing_combination) 
    compTensor = torch.stack((
        torch.tensor(sparseCompMat.row).long(),
        torch.tensor(sparseCompMat.col).long(),
        torch.tensor(sparseCompMat.data).long(),
    ))

    # Put on GPU if available
    if useCuda:
        compTensor = compTensor.cuda()

    # Compute the scores distributions for all the images (stimulis)
    scoreNormalDistList = asapGPU.true_skill(compTensor, rows)
    
    return scoreNormalDistList

def JODScores(compMat, useCuda=False, shiftToRefImage=False):
    
    scoreNormalDistList = inferScores(compMat, useCuda)
    
    # Get the mean and std for every image
    meanList = scoreNormalDistList.mean.cpu().detach().numpy().tolist()
    stdList = np.sqrt(scoreNormalDistList.variance.cpu().detach().numpy()).tolist()
    
    # Get the normalized JOD scores
    JOD, JODstd = normalizeScale(meanList, stdList, shiftToRefImage)
    
    return JOD, JODstd

def normalizeScale(score, scoreStd=None, shiftToRefImage=False):
    
    """ since the JND score is expressed by : 
        Score2 - Score1 = SIGMA_JOD * INV_NORMAL_DICT(P(R2>R1)), 
        we must multiply the scores by SIGMA_JOD to get the right scaling in JOD.
        Since the True_Skill gives us the score in a different scaling, we must:
            - first normalize, to align ourselves with the inverse function of a normal distribution;
            - Then, multiply by 1.4826 to get 1 JOD = 1
    """
    normalizedScore = (score - np.mean(score))/np.std(score) * SIGMA_JOD
    if shiftToRefImage:
        # JOD is relative ==> we suppose r1=0
        normalizedScore -= normalizedScore[0]
    if scoreStd:
        normalizedScorestd = scoreStd/np.std(score) * SIGMA_JOD
        return normalizedScore, normalizedScorestd
    return normalizedScore
    
    
############################################################
#oRIGINAL cODE
############################################################
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:29:32 2022

@author: prince
"""

import json
import numpy as np
import gluonnlp as nlp
import argparse
from WeatCode import compute_weat_score
from utils import *


def loadDataVocab(embedType, preTrainFile):
    fasttext_2M300d = nlp.embedding.create(embedType, source=preTrainFile)
    vocab = nlp.Vocab(nlp.data.Counter(fasttext_2M300d.idx_to_token))
    vocab.set_embedding(fasttext_2M300d)
    count_tok = nlp.data.Counter(fasttext_2M300d.idx_to_token)
    wordsVocab = [x[0] for x in count_tok.most_common()]
    return vocab, wordsVocab[1:]

def read_terms_from_file(file_path):
    with open(file_path, 'r') as f:
        terms = json.load(f)
    return terms

def Weat_dotprodResult(emb, gendered_male, gendered_female, name_male, name_female, pleasant, unpleasant):
    # Embeddings for gendered terms (e.g., 'he', 'him')
    gendered_male_emb = get_vecs(emb, gendered_male)
    gendered_female_emb = get_vecs(emb, gendered_female)

    # Embeddings for name terms (e.g., male names, female names)
    name_male_emb = get_vecs(emb, name_male)
    name_female_emb = get_vecs(emb, name_female)

    # Embeddings for pleasant and unpleasant terms
    pleasant_emb = get_vecs(emb, pleasant)
    unpleasant_emb = get_vecs(emb, unpleasant)

    # Calculate bias directions for gendered terms, name terms, and pleasant/unpleasant terms
    gendered_terms_direction, _, _ = bias_two_means(gendered_male_emb, gendered_female_emb)
    name_terms_direction, _, _ = bias_two_means(name_male_emb, name_female_emb)
    pleasant_unpleasant_direction, _, _ = bias_two_means(pleasant_emb, unpleasant_emb)

    # Compute the WEAT scores for different comparisons
    weat_score_gender_vs_names = compute_weat_score(emb, gendered_male, gendered_female, name_male, name_female)
    weat_score_gender_vs_pleasant_unpleasant = compute_weat_score(emb, gendered_male, gendered_female, pleasant, unpleasant)
    weat_score_names_vs_pleasant_unpleasant = compute_weat_score(emb, name_male, name_female, pleasant, unpleasant)

    # Print the results
    print("########################################")
    print("Dot Product During Training: Gender Term vs Gender Names ", np.round(np.dot(gendered_terms_direction, name_terms_direction), decimals=4))
    print("Dot Product During Training: Gender Term vs Plea/Unpleas ", np.round(np.dot(gendered_terms_direction, pleasant_unpleasant_direction), decimals=4))
    print("Dot Product During Training: Gender Names vs Plea/Unpleas ", np.round(np.dot(name_terms_direction, pleasant_unpleasant_direction), decimals=4))
    print()
    print("Weat Score: Gender Term vs Gender Names ", np.round(weat_score_gender_vs_names, decimals=4))
    print("Weat Score: Gender Term vs Plea/Unpleas ", np.round(weat_score_gender_vs_pleasant_unpleasant, decimals=4))
    print("Weat Score: Gender Names vs Plea/Unpleas ", np.round(weat_score_names_vs_pleasant_unpleasant, decimals=4))
    print()


###############################################################################
###############################################################################

def OSCaRSpan(emb, Vocabs, pleasant_emb, unpleasant_emb, sensitivity_emb, 
              gendered_male, gendered_female, gen_words, spanMatrix, 
              name_male, name_female, pleasant, unpleasant, iteration):
    
    span_matrix = []

        
    ######################################################
    #1. Get v1 & v2
    ######################################################
    v2, vec1_mean_v2, vec2_mean_v2 =  bias_two_means(pleasant_emb, unpleasant_emb)
    v1  =  closest_vec_span(v2, spanMatrix)
    

    ######################################################
    #3. Check for Angle Change v1 and v2
    ######################################################
    
    theta = angle_between(v1, v2)
    if theta > np.pi / 2:
        v2 = -v2   
    
    vecs = vectors(emb)
    rot_matrix = gsConstrained(np.identity(v1.shape[0]), v1, v2)   
    proj_newBasis = np.matmul(vecs, rot_matrix.T)
    
    x_coord = proj_newBasis[:, 0]
    y_coord = proj_newBasis[:, 1]

    ######################################################
    #4. Compute v2_prime
    ######################################################
    
    v2_prime = v2 - v1 * (v2.dot(v1))
    v2_prime = v2_prime / np.linalg.norm(v2_prime)
    
    ######################################################
    # Get Span Matrix
    ######################################################

    span_matrix.append(v1)
    span_matrix.append(v2_prime)
    
    ######################################################        
    #5. Update all K-d points to 2-d
    ######################################################
    
    V1_direction = np.array([v1.dot(v1), v1.dot(v2_prime)])
    V1_direction = V1_direction / np.linalg.norm(V1_direction)
    
    V2_direction = np.array([v2.dot(v1), v2.dot(v2_prime)])
    V2_direction = V2_direction / np.linalg.norm(V2_direction)
    
    emb =  dict(zip(Vocabs, np.vstack([x_coord, y_coord]).T)) 
    
    bias_direction = np.array([v1.dot(v1), v1.dot(v2_prime)])
    bias_direction = bias_direction / np.linalg.norm(bias_direction)
    orth_direction = np.array([v2.dot(v1), v2.dot(v2_prime)])
    orth_direction = orth_direction / np.linalg.norm(orth_direction)
    orth_direction_prime = np.array([v2_prime.dot(v1), v2_prime.dot(v2_prime)])
    orth_direction_prime = orth_direction_prime / np.linalg.norm(orth_direction_prime)
                
    ######################################################        
    #7. Centering
    ######################################################   
    
    midpoint = (vec1_mean_v2 + vec2_mean_v2) / 2
    point_inter = np.array([midpoint.dot(v1), midpoint.dot(v2_prime)])
    
    vectors_base = vectors(emb)
    
    normed_base_vectors = (vectors_base - point_inter)
    emb =  dict(zip(Vocabs, normed_base_vectors)) 
    
    ######################################################        
    #8. Do Correction
    ###################################################### 
                
    corrected_2d = []
    emb_2d = []
    gender_direction_2d = []
    occupation_direction_2d = []
    occupation_direction_2d_prime = []
    
    def doCorrection():
        for idx, wv in enumerate(vectors(emb)):
            x, dir1, dir2,  dir2_prime, rotated_head  = correction2d_new(rot_matrix, bias_direction, orth_direction, wv)
            emb_2d.append(x)
            gender_direction_2d.append(dir1)
            occupation_direction_2d.append(dir2)
            corrected_2d.append(rotated_head)
            occupation_direction_2d_prime.append(dir2_prime)

    doCorrection()
    
    
    rotated_head = np.array(corrected_2d)
    rotated_head = dict(zip(Vocabs, rotated_head)) 
    
    rotated_head_allwords = vectors(rotated_head) + point_inter
    
    proj_newBasis[:, :2] = rotated_head_allwords
    
    rotated_head_allD = np.matmul(proj_newBasis, rot_matrix)
    rotated_head_allD = dict(zip(Vocabs, rotated_head_allD)) 
        
    
    emb = rotated_head_allD.copy()
    
    pleasant_emb = get_vecs(emb, pleasant)
    unpleasant_emb = get_vecs(emb, unpleasant)
 
    print("########################################")
    print("Applying ISR Iteration " + str(iteration+1))
    ##########
    #Print out Dotproduct
    Weat_dotprodResult(emb, gendered_male, gendered_female, 
                      name_male, name_female, pleasant, unpleasant)
        ##########
    return emb, span_matrix

###############################################################################
###############################################################################

def OSCaRPairwise(emb, Vocabs, gendered_male_emb, gendered_female_emb, gen_emb,  
                  name_male_emb, name_female_emb, name_emb,
                  gendered_male, gendered_female, gen_words, name_male, name_female, name_words, 
                  pleasant, unpleasant, iteration):
    
    span_matrix = []
    
    # for iteration in range(1):
    #     print("Iteration " + str(iteration))
        
    ######################################################
    #1. Get v1 & v2
    ######################################################
    
    v1, vec1_mean_v1, vec2_mean_v1 =  bias_two_means(gendered_male_emb, gendered_female_emb)
    v2, vec1_mean_v2, vec2_mean_v2 =  bias_two_means(name_male_emb, name_female_emb)
    
    ##########
    #Print out Dotproduct
    if iteration == 0:
        print("########################################")
        print("Original Embedding without Debiasing")
        Weat_dotprodResult(emb, gendered_male, gendered_female, 
                          name_male, name_female, pleasant, unpleasant)
    ##########
    
    
    ######################################################
    #3. Check for Angle Change v1 and v2
    ######################################################
    
    theta = angle_between(v1, v2)
    if theta > np.pi / 2:
        v2 = -v2   
    
    
    vecs = vectors(emb)
    rot_matrix = gsConstrained(np.identity(v1.shape[0]), v1, v2)   
    proj_newBasis = np.matmul(vecs, rot_matrix.T)
    
    x_coord = proj_newBasis[:, 0]
    y_coord = proj_newBasis[:, 1]
        
    ######################################################
    #4. Compute v2_prime
    ######################################################
    
    v2_prime = v2 - v1 * (v2.dot(v1))
    v2_prime = v2_prime / np.linalg.norm(v2_prime)
    
    ######################################################
    # Get Span Matrix
    ######################################################
    # if iteration == 0:
    span_matrix.append(v1)
    span_matrix.append(v2_prime)
        
    ######################################################        
    #5. Update all K-d points to 2-d
    ######################################################
    
    V1_direction = np.array([v1.dot(v1), v1.dot(v2_prime)])
    V1_direction = V1_direction / np.linalg.norm(V1_direction)
    
    V2_direction = np.array([v2.dot(v1), v2.dot(v2_prime)])
    V2_direction = V2_direction / np.linalg.norm(V2_direction)
    
    emb =  dict(zip(Vocabs, np.vstack([x_coord, y_coord]).T)) 
    
    bias_direction = np.array([v1.dot(v1), v1.dot(v2_prime)])
    bias_direction = bias_direction / np.linalg.norm(bias_direction)
    orth_direction = np.array([v2.dot(v1), v2.dot(v2_prime)])
    orth_direction = orth_direction / np.linalg.norm(orth_direction)
    orth_direction_prime = np.array([v2_prime.dot(v1), v2_prime.dot(v2_prime)])
    orth_direction_prime = orth_direction_prime / np.linalg.norm(orth_direction_prime)
                
    ######################################################        
    #7. Centering
    ######################################################   
    
    midpoint = (vec1_mean_v2 + vec2_mean_v2) / 2
    point_inter = np.array([midpoint.dot(v1), midpoint.dot(v2_prime)])
    
    vectors_base = vectors(emb)
    
    normed_base_vectors = (vectors_base - point_inter)
    emb =  dict(zip(Vocabs, normed_base_vectors)) 
    
    ######################################################        
    #8. Do Correction
    ###################################################### 
                
    corrected_2d = []
    emb_2d = []
    gender_direction_2d = []
    occupation_direction_2d = []
    occupation_direction_2d_prime = []
    
    def doCorrection():
        for idx, wv in enumerate(vectors(emb)):
            x, dir1, dir2,  dir2_prime, rotated_head  = correction2d_new(rot_matrix, bias_direction, orth_direction, wv)
            emb_2d.append(x)
            gender_direction_2d.append(dir1)
            occupation_direction_2d.append(dir2)
            corrected_2d.append(rotated_head)
            occupation_direction_2d_prime.append(dir2_prime)

    doCorrection()
    
    
    rotated_head = np.array(corrected_2d)
    rotated_head = dict(zip(Vocabs, rotated_head)) 
    
    rotated_head_allwords = vectors(rotated_head) + point_inter
    
    proj_newBasis[:, :2] = rotated_head_allwords
    
    rotated_head_allD = np.matmul(proj_newBasis, rot_matrix)
    rotated_head_allD = dict(zip(Vocabs, rotated_head_allD)) 
    
    emb = rotated_head_allD.copy()
    
    gen_emb = get_vecs(emb, gen_words)
    gendered_male_emb = get_vecs(emb, gendered_male)
    gendered_female_emb = get_vecs(emb, gendered_female)
    
    name_emb = get_vecs(emb, name_words)
    name_male_emb = get_vecs(emb, name_male)
    name_female_emb = get_vecs(emb, name_female)
        
    return emb, span_matrix



###############################################################################

def loadDataVocab(embedType, preTrainFile):
    
    fasttext_2M300d = nlp.embedding.create(embedType, source=preTrainFile) 

    # create vocabulary by using all the tokens
    vocab = nlp.Vocab(nlp.data.Counter(fasttext_2M300d.idx_to_token))
    vocab.set_embedding(fasttext_2M300d) 
    #len(vocab.idx_to_token)
    count_tok = nlp.data.Counter(fasttext_2M300d.idx_to_token)
    wordsVocab = [x[0] for x in count_tok.most_common()]

    return vocab, wordsVocab[1:]

# nlp.embedding.list_sources('glove')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--embedType', type=str, default="glove")
    parser.add_argument('--preTrainFile', type=str, default="glove.6B.300d")
    parser.add_argument('--top_vocab', type=int, default=200000)
    parser.add_argument('--gendered_male_file', type=str,  required=True)
    parser.add_argument('--gendered_female_file', type=str, required=True)
    parser.add_argument('--name_male_file', type=str, required=True)
    parser.add_argument('--name_female_file', type=str,  required=True)
    parser.add_argument('--pleasant_file', type=str,  required=True)
    parser.add_argument('--unpleasant_file', type=str,  required=True)
    parser.add_argument('--output_file', type=str, required=True)

    args = parser.parse_args()
    
    embedType, preTrainFile = args.embedType, args.preTrainFile
    vocab, all_wordsVocab = loadDataVocab(embedType, preTrainFile)
    all_wordsVocab = all_wordsVocab[:args.top_vocab]

    base_emb = vocab.embedding[all_wordsVocab].asnumpy()
    base_emb = dict(zip(all_wordsVocab, base_emb))


############################################
    
    gendered_male = read_terms_from_file(args.gendered_male_file)
    gendered_female = read_terms_from_file(args.gendered_female_file)
    name_male = read_terms_from_file(args.name_male_file)
    name_female = read_terms_from_file(args.name_female_file)
    pleasant = read_terms_from_file(args.pleasant_file)
    unpleasant = read_terms_from_file(args.unpleasant_file)

    for itern in range(args.iterations):
        
        
        gen_words = gendered_male + gendered_female
        name_words = name_male + name_female

        gen_emb = get_vecs(base_emb, gen_words)
        gendered_male_emb = get_vecs(base_emb, gendered_male)
        gendered_female_emb = get_vecs(base_emb, gendered_female)

        name_emb = get_vecs(base_emb, name_words)
        name_male_emb = get_vecs(base_emb, name_male)
        name_female_emb = get_vecs(base_emb, name_female)
        
         
        base_emb, SpanMatrix = OSCaRPairwise(base_emb, all_wordsVocab, gendered_male_emb, gendered_female_emb, gen_emb,  
                          name_male_emb, name_female_emb, name_emb,
                          gendered_male, gendered_female, gen_words, name_male, name_female, name_words, 
                          pleasant, unpleasant, itern)
         
    
    ###############################################################################
        
        sensitivity = pleasant + unpleasant
        
        sensitivity_emb = get_vecs(base_emb, sensitivity)
        pleasant_emb = get_vecs(base_emb, pleasant)
        unpleasant_emb = get_vecs(base_emb, unpleasant)
        
        base_emb, _ = OSCaRSpan(base_emb, all_wordsVocab, pleasant_emb, unpleasant_emb, sensitivity_emb, 
                      gendered_male, gendered_female, gen_words, SpanMatrix, 
                      name_male, name_female, pleasant, unpleasant, itern)
    
    
    ######## Save Debiased Embedding
    saveEmbed(output_file, all_wordsVocab, vectors(base_emb) ) 
    ###############################################################################
    
     
     
        
        

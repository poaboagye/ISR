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

def ISR(iteration, gender_emb1, gender_emb2, occupation_emb1, occupation_emb2, base_emb, all_wordsVocab, gender_words, occupation_words, X, Y, A, B):
    print("##################################")
    print("Iteration " + str(iteration+1))
    
    v1, vec1_mean_v1, vec2_mean_v1 =  bias_two_means(gender_emb1, gender_emb2)
    v2, vec1_mean_v2, vec2_mean_v2 =  bias_two_means(occupation_emb1, occupation_emb2)
    
    
    theta = angle_between(v1, v2)
    if theta > np.pi / 2:
        v2 = -v2   

    vecs = vectors(base_emb)
    rot_matrix = gsConstrained(np.identity(v1.shape[0]), v1, v2)   
    proj_newBasis = np.matmul(vecs, rot_matrix.T)
    
    x_coord = proj_newBasis[:, 0]
    y_coord = proj_newBasis[:, 1]
    
    ######################################################
    #Compute v2_prime
    ######################################################
    
    v2_prime = v2 - v1 * (v2.dot(v1))
    v2_prime = v2_prime / np.linalg.norm(v2_prime)
        
    ######################################################        
    #Update all K-d points to 2-d
    ######################################################
    
    V1_direction = np.array([v1.dot(v1), v1.dot(v2_prime)])
    V1_direction = V1_direction / np.linalg.norm(V1_direction)
    
    V2_direction = np.array([v2.dot(v1), v2.dot(v2_prime)])
    V2_direction = V2_direction / np.linalg.norm(V2_direction)
    
    base_emb =  dict(zip(all_wordsVocab, np.vstack([x_coord, y_coord]).T)) 
    
    bias_direction = np.array([v1.dot(v1), v1.dot(v2_prime)])
    bias_direction = bias_direction / np.linalg.norm(bias_direction)
    orth_direction = np.array([v2.dot(v1), v2.dot(v2_prime)])
    orth_direction = orth_direction / np.linalg.norm(orth_direction)
    orth_direction_prime = np.array([v2_prime.dot(v1), v2_prime.dot(v2_prime)])
    orth_direction_prime = orth_direction_prime / np.linalg.norm(orth_direction_prime)
                
    ######################################################               
    #Centering
    ######################################################   
    
    midpoint = (vec1_mean_v2 + vec2_mean_v2) / 2
    point_inter = np.array([midpoint.dot(v1), midpoint.dot(v2_prime)])
    
    vectors_base = vectors(base_emb)
    
    normed_base_vectors = (vectors_base - point_inter)
    base_emb =  dict(zip(all_wordsVocab, normed_base_vectors)) 
    
    ######################################################        
    #Do Correction
    ###################################################### 
                
    corrected_2d = []
    emb_2d = []
    gender_direction_2d = []
    occupation_direction_2d = []
    occupation_direction_2d_prime = []
    
    def doCorrection():
        for idx, wv in enumerate(vectors(base_emb)):
            x, dir1, dir2,  dir2_prime, rotated_head  = correction2d_new(rot_matrix, bias_direction, orth_direction, wv)
            emb_2d.append(x)
            gender_direction_2d.append(dir1)
            occupation_direction_2d.append(dir2)
            corrected_2d.append(rotated_head)
            occupation_direction_2d_prime.append(dir2_prime)

    doCorrection()
    
    xx = np.array(emb_2d)
    xx = dict(zip(all_wordsVocab, xx)) 
    xx =  get_vecs(xx, gender_words + occupation_words)
    
    rotated_head = np.array(corrected_2d)
    rotated_head = dict(zip(all_wordsVocab, rotated_head)) 
    
    rotated_head_allwords = vectors(rotated_head) + point_inter
    
    rotated_head =  get_vecs(rotated_head, gender_words + occupation_words)
    
    v11 = np.array(gender_direction_2d[0])
    v22 = np.array(occupation_direction_2d[0])
    v2_pp = np.array(occupation_direction_2d_prime[0])
     

    proj_newBasis[:, :2] = rotated_head_allwords
    rotated_head_allD = np.matmul(proj_newBasis, rot_matrix)
    rotated_head_allD = dict(zip(all_wordsVocab, rotated_head_allD)) 
    
    
    base_emb = rotated_head_allD.copy()

    print("Dot Product Scores " + str(np.dot(v1, v2)))
    v1, vec1_mean_v1, vec2_mean_v1 =  bias_two_means(gender_emb1, gender_emb2)
    v2, vec1_mean_v2, vec2_mean_v2 =  bias_two_means(occupation_emb1, occupation_emb2)
    
    wt_score = compute_weat_score(base_emb, X, Y, A, B)
    print("Weat Scores is " + str(np.round(np.abs(wt_score), decimals = 4)))
    print()
    
    gender_emb = get_vecs(base_emb, gender_words)
    gender_emb1 = get_vecs(base_emb, X)
    gender_emb2 = get_vecs(base_emb, Y)
    
    occupation_emb = get_vecs(base_emb, occupation_words)
    occupation_emb1 = get_vecs(base_emb, A)
    occupation_emb2 = get_vecs(base_emb, B)
    
    return wt_score, np.dot(v1, v2), base_emb, gender_emb1, gender_emb2, occupation_emb1, occupation_emb2

def calculate_WeatDotprod_scores(iterations=10, embedType="glove", preTrainFile="glove.6B.300d", top_vocab=200000, X_file=None, Y_file=None, A_file=None, B_file=None, output_file=None):
    vocab, all_wordsVocab = loadDataVocab(embedType, preTrainFile)
    all_wordsVocab = all_wordsVocab[:top_vocab]

    base_emb = vocab.embedding[all_wordsVocab].asnumpy()
    base_emb = dict(zip(all_wordsVocab, base_emb))

    X = read_terms_from_file(X_file)
    Y = read_terms_from_file(Y_file)
    A = read_terms_from_file(A_file)
    B = read_terms_from_file(B_file)

    gender_words = X + Y
    occupation_words = A + B

    gender_emb = get_vecs(base_emb, gender_words)
    gender_emb1 = get_vecs(base_emb, X)
    gender_emb2 = get_vecs(base_emb, Y)

    occupation_emb = get_vecs(base_emb, occupation_words)
    occupation_emb1 = get_vecs(base_emb, A)
    occupation_emb2 = get_vecs(base_emb, B)
    
    print()
    print("##################################") 
    print("Original Embedding without Debiasing")
    v1, vec1_mean_v1, vec2_mean_v1 =  bias_two_means(gender_emb1, gender_emb2)
    v2, vec1_mean_v2, vec2_mean_v2 =  bias_two_means(occupation_emb1, occupation_emb2)
    print("Dot Product Scores " + str(np.dot(v1, v2)))
    
    wt_score = compute_weat_score(base_emb, X, Y, A, B)
    print("Weat Scores is " + str(np.round(wt_score, decimals = 4)))
    print()

    result_weat = []
    dotProd = []
    
    for iteration in range(iterations):
        wt_score, dp, base_emb, gender_emb1, gender_emb2, occupation_emb1, occupation_emb2 = ISR(
            iteration, gender_emb1, gender_emb2, occupation_emb1, occupation_emb2, base_emb, all_wordsVocab, gender_words, occupation_words, X, Y, A, B)
        
        result_weat.append(np.abs(wt_score))
        dotProd.append(np.abs(dp))

    print("################# Weat Scores for All Iterations #################")      
    result_weat = np.array(result_weat)
    print(np.round(result_weat, decimals=4))
    print()

    print("################# Dot Product Scores for All Iterations #################")      
    dotProd = np.array(dotProd)
    print(np.round(np.abs(dotProd), decimals=4))
    print()

######## Save Debiased Embedding
    saveEmbed(output_file, all_wordsVocab, vectors(base_emb) ) 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--embedType', type=str, default="glove")
    parser.add_argument('--preTrainFile', type=str, default="glove.6B.300d")
    parser.add_argument('--top_vocab', type=int, default=200000)
    parser.add_argument('--X_file', type=str, required=True)
    parser.add_argument('--Y_file', type=str, required=True)
    parser.add_argument('--A_file', type=str, required=True)
    parser.add_argument('--B_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)

    args = parser.parse_args()

    calculate_WeatDotprod_scores(args.iterations, args.embedType, args.preTrainFile, args.top_vocab, args.X_file, args.Y_file, args.A_file, args.B_file, args.output_file)

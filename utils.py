import numpy as np
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')

import gluonnlp as nlp

########
def load_embed(filename, max_vocab=-1): 
    words, embeds = [], []
    with open(filename, 'r') as f:
        next(f)
        for line in f:
            word, vector = line.rstrip().split(' ', 1)
            vector = np.fromstring(vector, sep=' ')
            words.append(word)
            embeds.append(vector)
            if len(embeds) == max_vocab:
                break
    return words, np.array(embeds)

########
def saveEmbed(path, words, word_embeds):
    with open(path, 'w') as f:
        print(word_embeds.shape[0], word_embeds.shape[1], file=f)
        for word, embed in zip(words, word_embeds):
            vector_str = ' '.join(str(x) for x in embed)
            print(word, vector_str, file = f)
                
########
def get(word_vectors, word):
    """
    Get WordVector object for single word
    """
    return word_vectors[word]

########
def get_many(word_vectors, words):
    """
    Get list of WordVector objects for a list of words
    """
    return [word_vectors[word] for word in words]

########
def get_vecs(word_vectors, words):
    """
    Get numpy array of vectors for a given list of words
    """
    return np.vstack([word_vectors[word] for word in words])

########
def vectors(word_vectors):
    return np.vstack([wv for wv in word_vectors.values()])

########
def words(word_vectors):
    return [wv.word for wv in word_vectors.values()]

########
def update_vectors(words, new_vectors, word_vectors):
    for i, word in enumerate(words):
        word_vectors[word] = new_vectors[i]
            
########         
def remove_center(embeddings):
    center = embeddings.mean(axis=0)[np.newaxis, :]
    embeddings -= center 
    return center, embeddings

########
def bias_two_means(vec1, vec2 ):
    vec1_mean, vec2_mean = np.mean(vec1, axis=0), np.mean(vec2, axis=0)
    bias_direction = (vec1_mean - vec2_mean) / np.linalg.norm(vec1_mean - vec2_mean)

    return bias_direction / np.linalg.norm(bias_direction), vec1_mean / np.linalg.norm(vec1_mean), vec2_mean/ np.linalg.norm(vec2_mean)

########
def get_he_she_basis(emb):
    assert(len(emb.shape) == 2)
    he = emb[0]
    she = emb[1]
    basis = (he - she) / np.linalg.norm(he - she)
    return  basis

########
def get_basis(emb):
    assert(len(emb.shape) == 2)
    pca = PCA(n_components=2)
    pca.fit(emb)
    direction_vector = pca.components_[0]
    return direction_vector / np.linalg.norm(direction_vector)

########
def proj(u, a):
    return (np.dot(u, a)) * u

########
def gsConstrained(matrix,v1,v2):
    v1 = np.asarray(v1).reshape(-1)
    v2 = np.asarray(v2).reshape(-1)
    u = np.zeros((np.shape(matrix)[0],np.shape(matrix)[1]))
    u[0] = v1
    u[0] = u[0]/np.linalg.norm(u[0])
    u[1] = v2 - proj(u[0],v2)
    u[1] = u[1]/np.linalg.norm(u[1])
    for i in range(0,len(matrix)-2):
        p = 0.0
        for j in range(0,i+2):    
            p = p + proj(u[j],matrix[i])
        u[i+2] = matrix[i] - p
        u[i+2] = u[i+2]/np.linalg.norm(u[i+2])
    return u

########
def basis(vec):
    first_component = vec[0]
    second_component = vec[1]
    v2_prime = second_component - first_component * float(np.matmul(first_component, second_component.T))
    v2_prime = v2_prime / np.linalg.norm(v2_prime)
    return v2_prime

########
def proj_new(vec):
    first_component = vec[0]
    second_component = vec[1]
    return first_component * float(np.matmul(first_component, second_component.T))

########
def rotation(v1, v2, x):
    input_vec = x.copy()
    v2P = v2 - proj(v1, v2)
    v2P = v2P / np.linalg.norm(v2P)

    thetaP = np.arccos(np.dot(v1, v2))
    theta = np.abs(thetaP - np.pi / 2)

    x_norm = x / np.linalg.norm(x)
    phi = np.arccos(np.dot(v1 / np.linalg.norm(v1), x_norm))
    d = np.dot(v2P, x_norm)

    if d > 0 and phi < thetaP:
        thetaX = theta * (phi / thetaP)
    elif d > 0 and phi > thetaP:
        thetaX = theta * ((np.pi - phi) / (np.pi - thetaP + 1e-10))
    elif d < 0 and phi >= np.pi - thetaP:
        thetaX = theta * ((np.pi - phi) / thetaP)
    elif d < 0 and phi < np.pi - thetaP:
        thetaX = theta * (phi / (np.pi - thetaP + 1e-10))
    else:
        return input_vec, v1, v2, v2P, x

    R = np.zeros((2, 2))
    R[0][0] = np.cos(thetaX)
    R[0][1] = -np.sin(thetaX)
    R[1][0] = np.sin(thetaX)
    R[1][1] = np.cos(thetaX)

    return input_vec, v1, v2, v2P, np.matmul(R, x)

########        
def correction2d_new(U, v1, v2, x):
    return rotation(v1, v2, x)
   
########
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

########
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


########
def load_wordList(filename): 
    my_file = open(filename, "r",  encoding="ISO-8859-1")
    data = my_file.read()
    data_into_list = data.replace(' ', '').split(",")  
    my_file.close()
    return data_into_list

########
def closest_vec_span(v, spanMatrix):
    v/=np.linalg.norm(v)
    spanMatrix[0] = spanMatrix[0]/np.linalg.norm(spanMatrix[0])
    spanMatrix[1] = spanMatrix[1]/np.linalg.norm(spanMatrix[1])
    num_proj_onto = 2
    closest_vec = 0.0
    for j in range(num_proj_onto):
        closest_vec = closest_vec + proj(spanMatrix[j], v)
    return closest_vec/ np.linalg.norm(closest_vec)


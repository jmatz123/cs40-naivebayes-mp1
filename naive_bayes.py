# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import numpy as np
import math
import re
import itertools
# from itertools import islice, izip
from tqdm import tqdm
from collections import Counter
import reader
 
"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
 
 
 
 
"""
 load_data calls the provided utility to load in the dataset.
 You can modify the default values for stemming and lowercase, to improve performance when
      we haven't passed in specific values for these parameters.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
   print(f"Stemming is {stemming}")
   print(f"Lowercase is {lowercase}")
   train_set, train_labels, dev_set, dev_labels = reader.load_dataset_main(trainingdir,testdir,stemming,lowercase,silently)
   return train_set, train_labels, dev_set, dev_labels
 
 
def create_word_maps_uni(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it
    
    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1
        keys: words
        values: number of times the word appears
    neg_vocab:
        In data where labels are 0
        keys: words
        values: number of times the word appears
    """
    #print(len(X),'X')
    pos_vocab = {}
    neg_vocab = {}
    pos_vocab_list = []
    neg_vocab_list = []
    
 
# train_set = a list of emails; each email is a list of words
#     train_labels = a list of labels, one label per email; each label is 1 or 0
   ##TODO:
#    raise RuntimeError("Replace this line with your code!")

    for i in range(len(X)) :
        for word in X[i] :
            if (y[i] == 1) :
                pos_vocab_list.append(word)
                #    print("++++++++++" + str(pos_vocab))
            else :
                neg_vocab_list.append(word)

    pos_vocab = Counter(pos_vocab_list)
    neg_vocab = Counter(neg_vocab_list)
 
    return dict(pos_vocab), dict(neg_vocab)


def create_word_maps_bi(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it
    
    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1
        keys: pairs of words
        values: number of times the word pair appears
    neg_vocab:
        In data where labels are 0
        keys: words
        values: number of times the word pair appears
    """
    #print(len(X),'X')
    pos_vocab = {}
    neg_vocab = {}
    pos_vocab_list = []
    neg_vocab_list = []
    ##TODO:
#    raise RuntimeError("Replace this line with your code!")
    for i in range(len(X)) :
        for word in X[i] :
            if (y[i] == 1) :
                pos_vocab_list.append(word)
                
            else :
                neg_vocab_list.append(word)

        if (y[i] == 1) :
            pos_output = [(k, m.split()[n + 1]) for m in pos_vocab_list for n, k in enumerate(m.split()) if n < len(m.split()) - 1]
        else :
            neg_output = [(k, m.split()[n + 1]) for m in neg_vocab_list for n, k in enumerate(m.split()) if n < len(m.split()) - 1]

    pos_vocab = Counter(pos_output)
    neg_vocab = Counter(neg_output)

    uni_pos_map, uni_neg_map = create_word_maps_uni(X, y, max_size = None)

    pos_vocab = pos_vocab | uni_pos_map
    neg_vocab = neg_vocab | uni_neg_map

    return dict(pos_vocab), dict(neg_vocab)

# Keep this in the provided template
def print_paramter_vals(laplace,pos_prior):
   print(f"Unigram Laplace {laplace}")
   print(f"Positive prior {pos_prior}")

"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=0.001, pos_prior=0.8, silently=False):

    # Keep this in the provided template
    print_paramter_vals(laplace,pos_prior)

    predicted_labels = []
    neg_prior = 1 - pos_prior

    result = create_word_maps_uni(train_set, train_labels, max_size=None)
    positives = result[0]
    negatives = result[1]

    pos_word_probs, no_word_pos = uni_probability_calculation(positives, laplace)
    neg_word_probs, no_word_neg = uni_probability_calculation(negatives, laplace)

    for email in dev_set : 
        positive_prob = 0
        negative_prob = 0
        for word in email : 

            if word in pos_word_probs :
                positive_prob += np.log(pos_word_probs[word])
            else :
                positive_prob += np.log(no_word_pos)
            
            if word in neg_word_probs :
                negative_prob += np.log(neg_word_probs[word])
            else :
                negative_prob += np.log(no_word_neg)

        positive_prob += pos_prior
        negative_prob += neg_prior
        # print(str(positive_prob) + " negative: " + str(negative_prob))

        if (negative_prob > positive_prob) :
            predicted_labels.append(0)
        else :
            predicted_labels.append(1)

        # print(predicted_labels[len(predicted_labels) - 1])
        
    return predicted_labels

            
    #    raise RuntimeError("Replace this line with your code!")
    
    #TODO figure out what this is supposed to return
    return []


def uni_probability_calculation(word_count_map, laplace) :
    probabilities = {}
    total_word_count= 0

    for word in word_count_map :
        total_word_count+= word_count_map[word]

    for word in word_count_map : 
        numerator = word_count_map[word] + laplace
        denominator = total_word_count + (laplace * (1 + len(word_count_map)))
        probability = numerator / denominator
        
        probabilities[word] = probability
    
    no_word_prob = (laplace) / (total_word_count + (laplace * (1 + len(word_count_map))))

    return probabilities, no_word_prob

def bi_probability_calculation(bigram_word_count_map, unigram_word_count_map, laplace) :
    probabilities = {}
    total_bigram_count = 0
    total_unigram_count = 0

    for bigram in bigram_word_count_map :
        total_bigram_count += bigram_word_count_map[bigram]
    
    for word in unigram_word_count_map :
        total_unigram_count += unigram_word_count_map[word]

    for bigram in bigram_word_count_map : 
        numerator = bigram_word_count_map[bigram] + laplace
        denominator = (total_bigram_count + total_unigram_count) + (laplace * (1 + len(bigram_word_count_map) + len(unigram_word_count_map)))
        probability = numerator / denominator
        
        probabilities[bigram] = probability
    
    no_word_prob = (laplace) / (total_bigram_count + (laplace * (1 + len(bigram_word_count_map))))

    return probabilities, no_word_prob


# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")
    
    
def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.001, bigram_laplace=0.005, bigram_lambda=0.5,pos_prior=0.8,silently=False):
    '''
    Compute a unigram+bigram naive Bayes model; use it to estimate labels on a dev set.
    
    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    unigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    bigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating bigram probs
    bigram_lambda (scalar float) = interpolation weight for the bigram model
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations
    
    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)
    
    max_vocab_size = None

    predicted_labels = []
    neg_prior = 1 - pos_prior
    unigram_lambda = 1 - bigram_lambda
    space = " "

    uni_positives, uni_negatives = create_word_maps_uni(train_set, train_labels, max_size=None)
    bi_positives, bi_negatives = create_word_maps_bi(train_set, train_labels, max_size=None)

    uni_pos_word_probs, uni_no_word_pos = uni_probability_calculation(uni_positives, unigram_laplace)
    uni_neg_word_probs, uni_no_word_neg = uni_probability_calculation(uni_negatives, unigram_laplace)

    bi_pos_word_probs, bi_no_word_pos = bi_probability_calculation(bi_positives, uni_positives, bigram_laplace)
    bi_neg_word_probs, bi_no_word_neg = bi_probability_calculation(bi_negatives, uni_negatives, bigram_laplace)

    for email in dev_set : 
        uni_positive_prob = 0
        uni_negative_prob = 0

        bi_positive_prob = 0
        bi_negative_prob = 0

        positive_prob = 0
        negative_prob = 0

        # bigram probs
        for bigram in email : 
            if space in bigram : 
                break

            if bigram in bi_pos_word_probs :
                bi_positive_prob += np.log(bi_pos_word_probs[bigram])
            else :
                bi_positive_prob += np.log(bi_no_word_pos)
            
            if bigram in bi_neg_word_probs :
                bi_negative_prob += np.log(bi_neg_word_probs[bigram])
            else :
                bi_negative_prob += np.log(bi_no_word_neg)

        for word in email :
            if word in uni_pos_word_probs :
                uni_positive_prob += np.log(uni_pos_word_probs[word])
            else :
                uni_positive_prob += np.log(uni_no_word_pos)
            
            if word in uni_neg_word_probs :
                uni_negative_prob += np.log(uni_neg_word_probs[word])
            else :
                uni_negative_prob += np.log(uni_no_word_neg)
        
        positive_prob = ((-1 * uni_positive_prob) ** unigram_lambda) * ((-1 * bi_positive_prob) ** bigram_lambda)
        positive_prob += pos_prior

        negative_prob = ((-1 * uni_negative_prob) ** unigram_lambda) * ((-1 * bi_negative_prob) ** bigram_lambda)
        negative_prob += neg_prior

        if (negative_prob < positive_prob) :
            predicted_labels.append(0)
        else :
            predicted_labels.append(1)
        
        
    return predicted_labels
    
 


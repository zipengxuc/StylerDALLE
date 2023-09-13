import torch
import os
import argparse
import json


def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)


def Nll(seq, gtSeq, returnScores=False):
    '''
    Compute the NLL loss of ground truth (target) sentence given the
    model. Assumes that gtSeq has <START> and <END> token surrounding
    every sequence and gtSeq is left aligned (i.e. right padded)

    S: <START>, E: <END>, W: word token, 0: padding token, P(*): logProb

        gtSeq:
            [ S     W1    W2  E   0   0]
        Teacher forced logProbs (seq):
            [P(W1) P(W2) P(E) -   -   -]
        Required gtSeq (target):
            [  W1    W2    E  0   0   0]
        Mask (non-zero tokens in target):
            [  1     1     1  0   0   0]
    '''
    gtLogProbs = torch.gather(seq, 2, gtSeq.unsqueeze(2)).squeeze(2)
    # nll_loss = -torch.mean(gtLogProbs)
    return gtLogProbs


def maskedNll(seq, gtSeq, returnScores=False):
    '''
    Compute the NLL loss of ground truth (target) sentence given the
    model. Assumes that gtSeq has <START> and <END> token surrounding
    every sequence and gtSeq is left aligned (i.e. right padded)

    S: <START>, E: <END>, W: word token, 0: padding token, P(*): logProb

        gtSeq:
            [ S     W1    W2  E   0   0]
        Teacher forced logProbs (seq):
            [P(W1) P(W2) P(E) -   -   -]
        Required gtSeq (target):
            [  W1    W2    E  0   0   0]
        Mask (non-zero tokens in target):
            [  1     1     1  0   0   0]
    '''
    gtLogProbs = torch.gather(seq, 2, gtSeq.unsqueeze(2)).squeeze(2)
    nll_loss = -torch.mean(gtLogProbs)
    return nll_loss

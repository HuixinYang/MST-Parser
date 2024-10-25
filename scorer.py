#!/usr/bin/env python3
""" Data Structures and Algorithms for CL III, Project
    See <https://dsacl3-2023.github.io/project/> for detailed instructions.

    Course:      Data Structures and Algorithms for CL III - WS2324
    Assignment:  Project
    Author:      Mario Kuzmanov, Huixin Yang
    Description: MST parser

    Honor Code:  I pledge that this program represents my own work.
"""

import numpy as np
from collections import Counter
from conllu import read_conllu


class BaselineScorer:
    """A simple baseline scoring class.

    train() method keeps two counters: one for childpos-headpos pairs,
    and another for childpos-deprel.

    score() returns the product of relevant counts for a given arc.
    """

    def __init__(self):
        self.deplabels = None
        self.heads = None

    def train(self, train_conllu, dev_conllu=None):

        self.deplabels = Counter()
        self.heads = Counter()
        for sent in read_conllu(train_conllu):
            for token in sent[1:]:
                self.deplabels[(token.upos, token.deprel)] += 1
                self.heads[(token.upos, sent[token.head].upos)] += 1

    def score(self, sent, i, j, deprel):
        if self.deplabels is None or self.heads is None:
            raise Exception("The model is not trained.")
        headpos, childpos = sent[i].upos, sent[j].upos
        return self.deplabels[(childpos, deprel)] \
            * self.heads[(childpos, headpos)]


class Scorer:
    # Assignment part 3: implement a better scorer below
    # You should keep the same interface: train() updates all
    # necessary information needed for the scoring, and score()
    # returns a score that is higher for high-probability edges.
    # You can implement any method you see fit. See the assignment
    # description for the requirements and some ideas that may help
    # implementing a simple scorer better than the baseline provided
    # above.
    #
    def __init__(self):
        self.freq_triplets = None
        self.dep_distance = None
        self.deprel_left = None
        self.deprel_right = None
        self.parent_deprel = None
        self.parent_plus_child_min = None
        self.parent_min_child_plus = None
        self.first = None
        self.last = None

    def train(self, train_conllu, dev_conllu=None):

        """
        Our consideration is dependency relation of a token could be predicted based on:
        - its postag, parent's postag;
        - parent's dependency label;
        - its index, especially it is at start or end of a sentence;
        - distance between its index and its parent's index in a sentence;
        - arc direction;
        - its form, feature;
        - context based on UPOS of +1 and -1 index token for both parent and child;
        """

        self.freq_triplets = Counter()
        self.dep_distance = Counter()
        self.deprel_left = Counter()
        self.deprel_right = Counter()
        self.parent_deprel = Counter()
        self.parent_plus_child_min = Counter()
        self.parent_min_child_plus = Counter()
        self.first = Counter()
        self.last = Counter()

        for sent in read_conllu(train_conllu):
            for i in range(1, len(sent)):

                # parent UPOS and child UPOS
                child_idx, parent_idx = i, sent[i].head
                child, parent = sent[child_idx], sent[parent_idx]
                parent_upos, child_upos = parent.upos, child.upos

                # child_dep based on child_upos and parent_upos;
                self.freq_triplets[(child_upos, child.deprel, parent_upos)] += 1

                # child_dep based on its index of sent(start or end), child_upos and parent_dep;
                if child_idx == 1:
                    self.first[(child_upos, parent.deprel, child.deprel)] += 1
                elif child_idx == len(sent) - 1:
                    self.last[(child.upos, parent.deprel, child.deprel)] += 1

                # child_dep based on distance based between parent index and child index;
                self.dep_distance[((abs(child_idx - parent_idx)), child.deprel)] += 1

                # child_dep based on arc direction;
                if child_idx < parent_idx:
                    self.deprel_left[child.deprel] += 1
                else:
                    self.deprel_right[child.deprel] += 1

                # child_dep based on word form, features and its relation to the parent
                self.parent_deprel[(child.form, child.feat, parent.deprel, child.deprel)] += 1

                # considering context based on UPOS of +1 and -1 index token for both parent and child
                parent_plus_1_upos = sent[parent_idx + 1].upos if parent_idx + 1 < len(sent) else None
                parent_min_1_upos = sent[parent_idx - 1].upos if parent_idx - 1 > 0 else None
                child_plus_1_upos = sent[child_idx + 1].upos if child_idx + 1 < len(sent) else None
                child_min_1_upos = sent[child_idx - 1].upos if child_idx - 1 > 0 else None
                self.parent_plus_child_min[(parent_upos, parent_plus_1_upos, child_min_1_upos, child_upos)] += 1
                self.parent_min_child_plus[(parent_min_1_upos, parent_upos, child_upos, child_plus_1_upos)] += 1

    def score(self, sent, i, j, deprel):
        """
        return: multiply the frequency from above counters;
        """
        if self.freq_triplets is None or self.dep_distance is None:
            raise Exception("The model is not trained.")

        # parent_upos, child_upos
        child_idx, parent_idx = j, i
        child, parent = sent[child_idx], sent[parent_idx]
        parent_upos, child_upos = parent.upos, child.upos

        # dep<- distance between child_index and parent_index;
        # dep<- child_upos, parent_upos
        prod = self.dep_distance[(abs(child_idx - parent_idx), deprel)] * self.freq_triplets[
            (child_upos, deprel, parent_upos)]

        # dep <- child_upos, parent_dep and its index;
        if child_idx == 1:
            prod *= self.first[(child_upos, parent.deprel, deprel)]
        elif child_idx == len(sent) - 1:
            prod *= self.last[(child_upos, parent.deprel, deprel)]

        # dep <- arc direction
        if parent_idx > child_idx:
            prod *= self.deprel_left[deprel]
        else:
            prod *= self.deprel_right[deprel]

        # dep <- form, feat, parent_dep
        if (child.form, child.feat, parent.deprel, deprel) in self.parent_deprel:
            prod *= self.parent_deprel[(child.form, child.feat, parent.deprel, deprel)]

        # context
        parent_plus_1_upos = sent[parent_idx + 1].upos if parent_idx + 1 < len(sent) else None
        parent_min_1_upos = sent[parent_idx - 1].upos if parent_idx - 1 >= 0 else None
        child_plus_1_upos = sent[child_idx + 1].upos if child_idx + 1 < len(sent) else None
        child_min_1_upos = sent[child_idx - 1].upos if child_idx - 1 >= 0 else None
        if (parent_upos, parent_plus_1_upos, child_min_1_upos, child_upos) in self.parent_plus_child_min:
            prod *= self.parent_plus_child_min[(parent_upos, parent_plus_1_upos, child_min_1_upos, child_upos)]
        if (parent_min_1_upos, parent_upos, child_upos, child_plus_1_upos) in self.parent_min_child_plus:
            prod *= self.parent_min_child_plus[(parent_min_1_upos, parent_upos, child_upos, child_plus_1_upos)]

        return prod
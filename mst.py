#!/usr/bin/env python3
""" Data Structures and Algorithms for CL III, Project
    See <https://dsacl3-2023.github.io/project/> for detailed instructions.

    Course:      Data Structures and Algorithms for CL III - WS2324
    Assignment:  Project
    Author:      Mario Kuzmanov, Huixin Yang
    Description: MST parser

    Honor Code:  I pledge that this program represents my own work.
"""
import os
import random

import numpy as np
import graphviz
import scorer
from conllu import read_conllu

UDREL = ("acl", "advcl", "advmod", "amod", "appos", "aux", "case",
         "cc", "ccomp", "clf", "compound", "conj", "cop", "csubj",
         "dep", "det", "discourse", "dislocated", "expl", "fixed", "flat",
         "goeswith", "iobj", "list", "mark", "nmod", "nsubj", "nummod",
         "obj", "obl", "orphan", "parataxis", "punct", "reparandum", "root",
         "vocative", "xcomp")


class DepGraph:
    """A directed graph implementation for MST parsing.

    """

    def __init__(self, sent, add_edges=True):
        self.nodes = [tok.copy() for tok in sent]
        n = len(self.nodes)
        self.M = np.zeros(shape=(n, n))
        self.deprels = [None] * n
        self.heads = [None] * n
        if add_edges:
            for i in range(1, n):
                self.add_edge(sent[i].head,
                              i, 1.0, sent[i].deprel)

    def add_edge(self, parent, child, weight=0.0, label="_"):
        self.M[parent, child] = weight
        self.deprels[child] = label
        self.nodes[child].head = parent
        self.nodes[child].deprel = label

    def remove_edge(self, parent, child, remove_deprel=True):
        self.M[parent, child] = 0.0
        if remove_deprel:
            self.deprels[child] = None

    def edge_list(self):
        """Iterate over all edges with non-zero weights.
        """
        for i in range(self.M.shape[0]):
            for j in range(self.M.shape[1]):
                if self.M[i, j] != 0.0:
                    yield (self.nodes[i], self.nodes[j],
                           self.M[i, j], self.deprels[j])

    def get_children(self, node):
        for i in range(self.M.shape[1]):
            if self.M[node, i] != 0.0:
                yield i, self.M[node, i], self.deprels[i]

    def get_parents(self, node):
        for i in range(self.M.shape[0]):
            if self.M[i, node] != 0.0:
                yield i, self.M[i, node], self.deprels[node]

    def _find_cycle(self, start=0):
        """Find a cycle from the start node using an interative DFS.
        """
        stack = [start]
        visited = {start: None}
        while stack:
            node = stack.pop()
            for child, _, _ in self.get_children(node):
                if child not in visited:
                    visited[child] = node
                    stack.append(child)
                else:
                    curr, path = node, [node]
                    while curr != start:
                        curr = visited[curr]
                        path.append(curr)
                    if child in path:
                        return list(reversed(path)), visited
                    visited[child] = node
                    stack.append(child)
        return [], visited

    def find_cycle(self):
        """Find and return a cycle if exists."""
        checked = set()
        for node in range(len(self.nodes)):
            if node in checked: continue
            cycle, visited = self._find_cycle(node)
            checked.update(set(visited))
            if cycle:
                return cycle
        return []

    def todot(self):
        """Return a GraphViz Digraph - can be useful for debugging."""
        dg = graphviz.Digraph()  # graph_attr={'rankdir': 'LR'})
        for head, dep, weight, deprel in self.edge_list():
            dg.edge(head.form, dep.form, label=f"{deprel}({weight:0.2f})")
        return dg


def mst_parse(sent, score_fn, deprels=UDREL):
    """Parse a given sentence with the MST algorithm.

    Note that even though we are doing labeled parsing, dependency
    labels are determined at the beginning (our scorers do not
    always assign the same score to the same label between two
    words)

    Parameters:
    sent: The input sentence represented as a sequence of Tokens.
    score_fn: A callable (function) that takes a sentence, the indices
              of parent and child nodes and a dependency label to
              assign a score to given graph edge (dependency arc).
              Note that larger the scores the better. We are
              interested in maximizing the total weight of tree edges,
              rather than minimizing.
    """
    n = len(sent)
    mst = DepGraph(sent, add_edges=False)

    # STEP 1: Find best head and deprel foreach child
    for child in range(1, n):
        maxscore, besthead, bestrel = 0.0, None, None
        for head in range(n):
            # to avoid unnecessary self-loop
            if child != head:
                for rel in deprels:
                    score = score_fn(sent, head, child, rel)
                    if score > maxscore:
                        maxscore, besthead, bestrel = score, head, rel
        mst.add_edge(besthead, child, maxscore, bestrel)

    # STEP 2: Check whether the resulting graph contains a cycle
    cycle = mst.find_cycle()

    # STEP 3: Break cycles until they exist in the graph
    removed = set()

    while len(cycle):
        minloss, bestu, bestv, oldp, bestw, bestrel = float('inf'), None, None, None, None, ""
        for v in cycle:
            # Root node doesn't have parents
            if v != 0:
                parent, _, _ = list(mst.get_parents(v))[0]
                deprel = mst.deprels[v]
                weight = score_fn(sent, parent, v, deprel)
                for u in range(n):
                    if u == v or u in cycle or (u, v) in removed:
                        continue
                    uw = score_fn(sent, u, v, deprel)
                    if weight - uw < minloss:
                        minloss = weight - uw
                        oldp = parent
                        bestu, bestv, bestw, bestrel = u, v, uw, deprel
        # Add the cyclic edge, for which we found min_loss replacement , to the removed set
        removed.add((oldp, bestv))
        # --------------------------
        # Problem1: Add the to-be-removed edge to remove set does not mean this edge is actually removed;
        # What the initial code does is add one additional edge rather than remove it;
        #
        # Problem2: In some languages it wouldn't be possible to refer to variables,
        # that's why some changes involve introducing scope variables
        mst.remove_edge(oldp, bestv)
        # --------------------------
        mst.add_edge(bestu, bestv, bestw, bestrel)
        cycle = list(mst.find_cycle())

    # STEP 4: Resulting graph is the MST
    return mst


def evaluate(gold_sent, pred_sent):
    """Calculate and return labeled and unlabeled attachment scores."""
    #
    # BUG 3/3: fix a trivial bug below that results in
    # wrong calculation of the evaluation metric(s).
    #
    assert len(gold_sent) == len(pred_sent)
    n = len(gold_sent) - 1
    uas, las = 0, 0
    for i, gold in enumerate(gold_sent[1:], start=1):
        pred = pred_sent[i]
        if gold.head == pred.head:
            uas += 1
            if gold.deprel == pred.deprel:
                las += 1
    return uas / n, las / n


def evaluate_models(train_file, test_file):
    # train models
    sc_baseline = scorer.BaselineScorer()
    sc_baseline.train(train_file)

    sc_scorer = scorer.Scorer()
    sc_scorer.train(train_file)

    # evaluate baseline vs. scorer
    uas_base, las_base, n = 0, 0, 0
    uas_scorer, las_scorer = 0, 0
    for sent in read_conllu(test_file):
        mst_baseline = mst_parse(sent, score_fn=sc_baseline.score)
        uas_sent, las_sent = evaluate(sent, mst_baseline.nodes)
        uas_base += uas_sent
        las_base += las_sent

        mst_scorer = mst_parse(sent, score_fn=sc_scorer.score)
        uas_sent2, las_sent2 = evaluate(sent, mst_scorer.nodes)
        uas_scorer += uas_sent2
        las_scorer += las_sent2

        n += 1

    # show results
    print("---------------------------------------------------------------------------------")
    print(f"BASELINE:\nUAS: {uas_base / n} LAS: {las_base / n}")
    print(f"SCORER:\nUAS: {uas_scorer / n} LAS: {las_scorer / n}")
    print("---------------------------------------------------------------------------------")


if __name__ == "__main__":
    # Example usage with a directory. In case it doesn't contain training set we split the testing set
    # Train and Test files are saved locally in ./data_splits

    import argparse

    from splitter import Splitter

    # ABSOLUTE PATH TO TREEBANK
    ap = argparse.ArgumentParser()
    ap.add_argument('treebank', help="Absolute path to directory that holds the UD treebank.")
    args = ap.parse_args()

    path = args.treebank

    # if you have own training and test files, just provide their paths here
    # and don't use command line
    train_file, test_file = Splitter(path).separate_files()

    evaluate_models(train_file, test_file)

    test_many = input("Do you want to test more treebanks? Y/N: ")

    if test_many.lower() == "y":
        # in case of OSError try changing the direction of back-slashes
        # or removing quotes from the path
        path = input("Absolute path to directory that holds latest UD release treebanks: ")
        treebanks_n = int(input("How many treebanks do you want to test?: "))
        # random choice of selected number of treebanks
        lst_random = random.sample(os.listdir(path), treebanks_n)
        for treebank_name in lst_random:
            print(f"TREEBANK: {treebank_name}")
            print("Loading...")
            path2 = f"{path}/{treebank_name}"
            train_file, test_file = Splitter(path2).separate_files()
            evaluate_models(train_file, test_file)
    else:
        exit()
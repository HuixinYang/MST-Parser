# -----------------------------------------------------------------------------------------------------------------------------------#
# TEST BETTER SCORER
# PLEASE NOTE IT TAKES SOME TIME TO RUN SINCE IT PROCESSES ALL AVAILABLE ENGLISH TREEBANKS (around 2-3min)
# THE TREEBANKS WE USED ARE ONLY THESE THAT HAVE SPLIT OF TRAIN AND TEST SETS
import pytest
from collections import defaultdict
import os
from mst import *

path = "treebank_test_cases/english_treebanks"


def evaluate_model(sc):
    """
       return: dictionary containing scores foreach treebank and total scores over all treebanks
               with weights generated from some scoring function
    """
    scores = defaultdict(list)
    lst = os.listdir(path)
    uas_english_treebanks, las_english_treebanks = 0, 0
    n_total = 0
    for dir_ in lst:
        test_file, train_file = os.listdir(f"{path}/{dir_}")
        test_file = f"{path}/{dir_}/{test_file}"
        train_file = f"{path}/{dir_}/{train_file}"

        sc.train(train_file)
        uassum, lassum, n = 0, 0, 0
        for sent in read_conllu(test_file):
            mst = mst_parse(sent, score_fn=sc.score)
            parsed = mst.nodes
            uas, las = evaluate(sent, parsed)
            uassum += uas
            lassum += las
            n += 1
        scores[dir_] = [("UAS", uassum / n), ("LAS", lassum / n)]
        uas_english_treebanks += uassum
        las_english_treebanks += lassum
        n_total += n
    scores["TOTALS"] = [("TOTAL UAS", uas_english_treebanks / n_total), ("TOTAL LAS", las_english_treebanks / n_total)]
    return scores


@pytest.mark.parametrize("old_scorer, our_scorer",
                         [
                             (evaluate_model(scorer.BaselineScorer()), evaluate_model(scorer.Scorer())),
                         ]
                         )
def test_scorer(old_scorer, our_scorer):
    # we expect both dictionaries to have exact same keys
    # since the treebanks are the same in both cases
    vals = []
    for treebank_name, (_uas_old, _las_old) in old_scorer.items():
        _uas_our, _las_our = our_scorer[treebank_name]
        if _uas_old[1] < _uas_our[1] and _las_old[1] < _las_our[1]:
            print(f"Treebank {treebank_name}. There is improvement in these treebanks.....")
            print(f"Old model UAS, LAS: {_uas_old[1], _las_old[1]}")
            print(f"Our model UAS, LAS: {_uas_our[1], _las_our[1]}")
            print("----------------------------------------------------------------------------------")
            vals.append(True)
        else:
            print(f"Treebank {treebank_name}. There is no improvement in some of these scores.....")
            print(f"Old model UAS, LAS: {_uas_old[1], _las_old[1]}")
            print(f"Our model UAS, LAS: {_uas_our[1], _las_our[1]}")
            print("----------------------------------------------------------------------------------")
            vals.append(False)
    assert False not in vals
# -----------------------------------------------------------------------------------------------------------------------------------#

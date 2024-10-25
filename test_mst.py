import mst
from mst import *
import conllu
import pytest
import scorer
import os
from collections import defaultdict


# Assignment part 2: write necessary tests that would at least detect
# the three bugs in the mst.py.
#
# Please try to make your tests concise and clearly understandable.
#
# -----------------------------------------------------------------------------------------------------------------------------------#
# TEST BUG 1
class DAG:

    def __init__(self, nodes):
        """
        This is the setup to a simpler class representing DAG
        It will be used to show what was the problem with original implementation
        """
        self.graph = []
        self.nodes = nodes

    def edges(self, edge_list):
        self.graph = edge_list

    def get_children(self, u):
        return [edge[1] for edge in self.graph if edge[0] == u]

    def _find_cycle(self, start=0):
        stack = [start]
        visited = {start: None}
        while stack:
            node = stack.pop()
            for child in self.get_children(node):
                if child not in visited:
                    visited[child] = node
                    stack.append(child)
                else:
                    curr, path = node, [node]
                    while curr != start:
                        curr = visited[curr]
                        path.append(curr)
                        # i = path.index(child)
                        # return list(reversed(path[:i + 1])), visited
                        if child in path:
                            return list(reversed(path)), visited
                    visited[child] = node
                    stack.append(child)
        return [], visited

    def find_cycle(self):
        checked = set()
        for node in range(len(self.nodes)):
            if node in checked: continue
            cycle, visited = self._find_cycle(node)
            checked.update(set(visited))
            if cycle:
                return cycle
        return ["_"]


@pytest.fixture(scope="session", autouse=True)
def testgraph():
    nodes = {0, 1, 2, 3, 4}
    return DAG(nodes)


@pytest.mark.parametrize("cycle, edge_list",
                         [
                             # here original implementation fails because there is no cycle
                             # since there is no cycle child != start and child will not be in the path
                             # trying to get its index will result in ValueError
                             ([["_"]], [(0, 1), (0, 2), (0, 4), (4, 2), (4, 1), (2, 3), (1, 3)]),
                             # --------------------------------------------------------------
                             ([[0, 4, 1, 3], [1, 3, 0], [4, 2, 3], [2, 3, 0]],
                              [(0, 1), (0, 2), (0, 4), (4, 2), (4, 1), (2, 3), (1, 3), (3, 0)]),

                             ([["_"]], [(0, 1), (1, 2), (2, 3), (3, 4)]),
                             ([[1, 2, 3, 4], [2, 3, 4, 1], [3, 4, 1, 2], [4, 1, 2, 3]],
                              [(1, 2), (2, 3), (3, 4), (4, 1)]),
                         ]
                         )
def test_graph(testgraph, cycle, edge_list):
    testgraph.edges(edge_list)
    assert testgraph.find_cycle() in cycle


# ------------------------------------------------------------------------------------------------------------------------------------#
# TEST BUG 2
# PLEASE NOTE THE PDF FILES IN ./outputs/mst_proper.
# THEY ARE SHOWING THE PROCESS OF FINDING AND REMOVING THE CYCLE WITH CORRECTED IMPLEMENTATION
# PDF FILES IN outputs/mst_initial HOWEVER SHOW WHAT THE INITIAL PARSING WAS DOING

train_file = "treebank_test_cases/english_treebanks/UD_English-Atis/en_atis-ud-train.conllu"
test_file = "treebank_test_cases/english_treebanks/UD_English-Atis/en_atis-ud-test.conllu"

# train a scoring function
sc = scorer.BaselineScorer()
sc.train(train_file)

# sentence to clearly show the problem with the initial mst parser
test_sent = list(read_conllu(test_file))[10]


def mst_parse_demo(sent, score_fn, deprels=UDREL, include_solution=False):
    """
    The following code is a slightly modified demo of parser;
    The idea here is to show with and without our modification, the parser product MST properly or not.
    """
    n = len(sent)
    mst = DepGraph(sent, add_edges=False)

    for child in range(1, n):
        maxscore, besthead, bestrel = 0.0, None, None
        for head in range(n):
            if child != head:
                for rel in deprels:
                    score = score_fn(sent, head, child, rel)
                    if score > maxscore:
                        maxscore, besthead, bestrel = score, head, rel
        mst.add_edge(besthead, child, maxscore, bestrel)

    cycle = mst.find_cycle()

    # Check whether we are removing cycles or not, but basically with the parser method in MST;
    cycles_detected = set()
    removed = set()
    while len(cycle):
        if tuple(cycle) in cycles_detected: return True
        cycles_detected.add((tuple(cycle)))
        minloss, bestu, bestv, oldp, bestw, bestrel = float('inf'), None, None, None, None, ""
        for v in cycle:
            lst_parent = list(mst.get_parents(v))
            if lst_parent:
                parent, _, _ = lst_parent[0]
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
        removed.add((oldp, bestv))
        if include_solution:
            mst.remove_edge(oldp, bestv)
        mst.add_edge(bestu, bestv, bestw, bestrel)
        cycle = list(mst.find_cycle())
    return False


@pytest.mark.parametrize("mst_parse, has_cycle",
                         [
                             (
                                     mst_parse_demo(test_sent, score_fn=sc.score, deprels=UDREL,
                                                    include_solution=False), True),
                             (
                                     mst_parse_demo(test_sent, score_fn=sc.score, deprels=UDREL, include_solution=True),
                                     False),
                         ]
                         )
def test_mst_parser(mst_parse, has_cycle):
    assert mst_parse == has_cycle


# ------------------------------------------------------------------------------------------------------------------------------------#
# TEST BUG 3

# read file and set up gold and predicted sentence
file = list(conllu.read_conllu("treebank_test_cases/hr_set-ud-train.conllu"))
gold, tested = file[:5], file[:5]


def generate_test_case(gold, tested, num=1):
    # Test 100% uas and 100% las
    if num == 1:
        return gold[0], tested[0]

    # Assume predict a wrong head and a correct label; will result not 100% uas and not 100% las.
    elif num == 2:
        new_predicted = []
        for token in tested[1]:
            if token.head == 2 and token.deprel == "nsubj":
                token = token.copy()
                token.head = 1
            new_predicted.append(token)
        return gold[1], new_predicted

    # Assume predict a correct head and a wrong label; will result 100% uas and not 100% las.
    elif num == 3:
        new_predicted = []
        for token in tested[2]:
            if token.deprel == "nsubj":
                token = token.copy()
                token.deprel = "csubj"
            new_predicted.append(token)
        return gold[2], new_predicted


@pytest.mark.parametrize("gold_output, predicted_output, score",
                         [
                             (generate_test_case(gold, tested)[0], generate_test_case(gold, tested)[1], (1.0, 1.0)),
                             # here the bug was found, if we had initial implementation we would have
                             # not full UAS score but full LAS since it was only comparing the deprel
                             (generate_test_case(gold, tested, num=2)[0], generate_test_case(gold, tested, num=2)[1],
                              (0.95, 0.95)),
                             (generate_test_case(gold, tested, num=3)[0], generate_test_case(gold, tested, num=3)[1],
                              (1.0, 0.95))
                         ]
                         )
def test_evaluation(gold_output, predicted_output, score):
    uas, las = evaluate(gold_output, predicted_output)
    assert pytest.approx(round(uas, 3), rel=1e-2) == score[0]
    assert pytest.approx(round(las, 3), rel=1e-2) == score[1]

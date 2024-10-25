
# MST Parser

A minimum spanning tree parser which labels the syntax relation between tokens with Universal Dependency (UD) arcs, such as labeling the sentence ["I", "like", "pizza"] as [nsubj, root, nobj]. Data structure and algorithm (WS 2023) assignment from [@MarioKuzmanov](https://github.com/MarioKuzmanov) and [@HuixinYang](https://github.com/HuixinYang)


## Part 1: MST

Minimum spanning tree based on Chu-Liu algorithm


## Part 2: Scorer

An improved arc scorer method without using any deep learning or NLP packages, trained and tested with _all English treebanks_ in the latest UD release (1.13).
- **Approach**: we extracted 7 linguistic features and used a kinda greedy approach to manage the trade off between the accuracy and computation load;
- **Accuracy**：we reached LAS (0.7), UAS (0.6) scores for compared to around 0.9 for RNN; 
- **Efficiency**: we could manage to train and validate the model for all the tree bank (around 760 k sentences in total) in around 2-4 min, compared with Hiden Markov or Word Embedding which take infinitely long time to run a single tree bank;

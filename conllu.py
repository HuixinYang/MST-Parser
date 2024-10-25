#!/usr/bin/env python3
import re

class Token:
    """A simple class for keeping information on UD syntactic tokens.
    """
    __slots__ = ('form', 'lemma', 'upos', 'feat', 'head', 'deprel')
    def __init__(self, form=None, lemma=None, upos=None, feat=None,
                 head=None, deprel=None):
        self.form = form
        self.lemma = lemma
        self.upos = upos
        self.feat = feat
        self.head = None if head is None or head == '_' else int(head)
        self.deprel = deprel

    def copy(self):
        """Return a copy of the current object.
        """
        return Token(form=self.form, lemma=self.lemma, upos=self.upos,
                     feat=self.feat, head=self.head, deprel=self.deprel)

def read_conllu(filename):
    """Read a CoNLL-U file, return a list of sentences.

    CoNLL-U files are the standard format for Universal Dependencies
    treebanks. They are simple text files that represent a sequence of
    sentences with morphological and syntactic annotations. Please see
    <https://universaldependencies.org/format.html> for the official
    reference. You do not need to know the exact specification of the
    CoNLL-U files, but it may be practical for writing some of the
    tests.

    This function reads (only) the information we may use for parsing
    from a CoNLL-U file, and yields each sentence (a list of 'Token's).
    """
    with open(filename, encoding="utf8") as f:
        sentence = [Token('<ROOT>')]
        for line in f:
            if len(line.strip()) == 0: # sentence boundary
                yield sentence
                sentence = [Token('<ROOT>')]
                continue
            if line.startswith('#'): # comments
                continue

            index, form, lemma, upos, _, feat,\
                head, deprel, _, _ = line.split('\t')
            if not re.match('^[0-9]+$', index): # not a basic dep. line
                continue
            sentence.append(Token(form=form, lemma=lemma, upos=upos,
                                  feat=feat, head=head, deprel=deprel))

# _*_ coding: utf-8 _*_

from __future__ import print_function
import nltk
from .util.preprocessing import addCharInformation, createMatrices, addCasingInformation
from .neuralnets.bilstm import BiLSTM
import sys


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python runModel.py modelPath inputPath")
        exit()

    modelPath = sys.argv[1]
    inputPath = sys.argv[2]

    # :: Read input ::
    with open(inputPath, 'r') as f:
        text = f.read()

    # :: Load the model ::
    lstmModel = BiLSTM.loadModel(modelPath)

    # :: Prepare the input ::
    sentences = [{'tokens': nltk.word_tokenize(sent)} for sent in nltk.sent_tokenize(text)]
    addCharInformation(sentences)
    addCasingInformation(sentences)
    dataMatrix = createMatrices(sentences, lstmModel.mappings, True)

    # :: Tag the input ::
    tags = lstmModel.tagSentences(dataMatrix)

    # :: Output to stdout ::
    for sentenceIdx in range(len(sentences)):
        tokens = sentences[sentenceIdx]['tokens']

        for tokenIdx in range(len(tokens)):
            tokenTags = []
            for modelName in sorted(tags.keys()):
                tokenTags.append(tags[modelName][sentenceIdx][tokenIdx])

            print("%s\t%s" % (tokens[tokenIdx], "\t".join(tokenTags)))
        print("")

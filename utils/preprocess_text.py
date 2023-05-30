# URLs:
# * preprocessing 1 : https://towardsdatascience.com/nlp-preprocessing-with-nltk-3c04ee00edc0
# * preprocessing 2 : https://www.nltk.org/api/nltk.tokenize.html

import string
from nltk.tokenize import LineTokenizer, sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
# Run the below line only the first time of running nltk
# nltk.download()

from .tokenize_text import tokenize_text

# Preprocess a text
def preprocess_text(text, glovemgr, labels_ner = None, is_sep_n = False, remove_stop_word = False, stemming=False, trunc_sent=-1, padding_sent=-1):
  # preprocess ner labels
  if labels_ner is not None:
    tokenized_flat_contents = tokenize_text(text)
    tmp_labels_ner = dict()

    for y in range(len(labels_ner)):
      for x in range(len(labels_ner[y])):
        tmp_labels_ner[tokenized_flat_contents[y][x]] = (0 if labels_ner[y][x] == 0 else 1)
  
    labels_ner = tmp_labels_ner

  # preprocess text
  result = tokenize_text(text=text, is_sep_n=is_sep_n, remove_stop_word=remove_stop_word, stemming=stemming)

  # trunc
  if trunc_sent >= 0:
    result = [line if len(line) <= trunc_sent else line[:trunc_sent] for line in result]

  # word2id
  result = [[glovemgr.w2i(word) for word in line] for line in result]

  # padding
  if padding_sent >= 0:
    result = [line + [0 for i in range(max(0, padding_sent - len(line)))] for line in result]

  if labels_ner is not None:
    return result, labels_ner
  
  return result

# URLs:
# * preprocessing 1 : https://towardsdatascience.com/nlp-preprocessing-with-nltk-3c04ee00edc0
# * preprocessing 2 : https://www.nltk.org/api/nltk.tokenize.html

import string
from nltk.tokenize import LineTokenizer, sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
import nltk
# Run the below line only the first time of running nltk
# nltk.download()

# Preprocess a text
def tokenize_text(text, is_sep_n = False, remove_stop_word = False, stemming=False):
  result = text

  # tokenize sentence
  if is_sep_n:
    nltk_line_tokenizer = LineTokenizer()
    result = nltk_line_tokenizer.tokenize(result)
  else:
    result = sent_tokenize(result, language="french")

  # lower
  result = [line.lower() for line in result]

  # Remove punctuation
  result = ["".join([char for char in line if char not in string.punctuation]) for line in result]

  # Tokenization
  result = [word_tokenize(line, language="french") for line in result]

  # Remove stop word
  if remove_stop_word:
    stop_words = stopwords.words('french')
    result = [[word for word in line if word not in stop_words] for line in result]

  # Stemming
  if stemming:
    porter = FrenchStemmer()
    result = [[porter.stem(word) for word in line] for line in result]

  return result
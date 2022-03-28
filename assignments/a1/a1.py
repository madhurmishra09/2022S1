import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
nltk.download('punkt')
nltk.download('gutenberg')
from sklearn.feature_extraction.text import TfidfVectorizer
import collections


from nltk.stem import PorterStemmer 



# Task 1 (2 marks)
import collections
def count_pos(document, pos):
    """Return the number of occurrences of words with a given part of speech. To find the part of speech, use 
    NLTK's "Universal" tag set. To find the words of the document, use NLTK's sent_tokenize and word_tokenize.
    >>> count_pos('austen-emma.txt', 'NOUN')
    31998
    >>> count_pos('austen-sense.txt', 'VERB')
    25074
    """

    """   stopwords = nltk.corpus.stopwords.words('english')
    
    """
    stopwords = nltk.corpus.stopwords.words('english')
    raw = nltk.corpus.gutenberg.raw(document)
    nltk.download('averaged_perceptron_tagger')
    import collections
    emma_sents = [nltk.word_tokenize(s) for s in nltk.sent_tokenize(nltk.corpus.gutenberg.raw('austen-emma.txt'))]
    emma_tagged_sents = [nltk.pos_tag(s, tagset="universal") for s in emma_sents]
    pos = []
    for s in emma_tagged_sents:
        for w in s:
            pos.append(w[1])
    counter = collections.Counter(pos)
    counter.most_common(5)
    return counter 

""""" comment start 
# Task 2 (2 marks)
def get_top_stem_bigrams(document, n): comment end """ 
    """Return the n most frequent bigrams of stems. Return the list sorted in descending order of frequency.
    The stems of words in different sentences cannot form a bigram. To stem a word, use NLTK's Porter stemmer.
    To find the words of the document, use NLTK's sent_tokenize and word_tokenize.
    >>> get_top_stem_bigrams('austen-emma.txt', 3)
    [(',', 'and'), ('.', "''"), (';', 'and')]
    >>> get_top_stem_bigrams('austen-sense.txt',4)
    [(',', 'and'), ('.', "''"), (';', 'and'), (',', "''")]
    """
  """ comment start  return []


# Task 3 (2 marks)
def get_same_stem(document, word): comment end """
    """Return the list of words that have the same stem as the word given, and their frequencies. 
    To find the stem, use NLTK's Porter stemmer. To find the words of the document, use NLTK's 
    sent_tokenize and word_tokenize. The resulting list must be sorted alphabetically.
    >>> get_same_stem('austen-emma.txt','respect')[:5]
    [('Respect', 2), ('respect', 41), ('respectability', 1), ('respectable', 20), ('respectably', 1)]
    >>> get_same_stem('austen-sense.txt','respect')[:5]
    [('respect', 22), ('respectability', 1), ('respectable', 14), ('respectably', 1), ('respected', 3)]
    """
   ''' sent_tokens = [word_tokenize(s) for s in sent_tokenize(nltk.corpus.gutenberg.raw(document))]
    stems= set()
    porter= nltk.PorterStemmer()
    for s in sent_tokens:
        stemmed= porter.stem(word)
        if stemmed == porter.stem(s):
            stems.add(s)
        else:
            return[0,0]
    return [stems,0]

# Task 4 (2 marks)
def most_frequent_after_pos(document, pos):
    """Return the most frequent word after a given part of speech, and its frequency. Do not consider words
    that occur in the next sentence after the given part of speech.
    To find the part of speech, use NLTK's "Universal" tagset.
    >>> most_frequent_after_pos('austen-emma.txt','VERB')
    [('not', 1932)]
    >>> most_frequent_after_pos('austen-sense.txt','NOUN')
    [(',', 5310)]
    """
    return []

# Task 5 (2 marks)
def get_word_tfidf(text):
    """Return the tf.idf of the words given in the text. If a word does not have tf.idf information or is zero, 
    then do not return its tf.idf. The reference for computing tf.idf is the list of documents from the NLTK 
    Gutenberg corpus. To compute the tf.idf, use sklearn's TfidfVectorizer with the option to remove the English 
    stop words (stop_words='english'). The result must be a list of words sorted in alphabetical order, together 
    with their tf.idf.
    >>> get_word_tfidf('Emma is a respectable person')
    [('emma', 0.8310852062844262), ('person', 0.3245184217533661), ('respectable', 0.4516471784898886)]
    >>> get_word_tfidf('Brutus is a honourable person')
    [('brutus', 0.8405129362379974), ('honourable', 0.4310718596448824), ('person', 0.32819971943754456)]
    """


    return [] 
    '''

# DO NOT MODIFY THE CODE BELOW
if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
    

import nltk
import PyPDF2
import string
import math
from collections import defaultdict
from ordered_set import OrderedSet

documents_count = 0

# Create a dictionary to store the total words in each documents
documents_words_num = defaultdict(int)

# Create a dictionary to store the term frequencies (TF)
tf = defaultdict(lambda: defaultdict(float))

# Create a dictionary to store the document frequencies (DF)
df = defaultdict(OrderedSet)

# Create a dictionary to store the TF-IDF scores
tfidf = defaultdict(lambda: defaultdict(float))


def tf_create(number_of_words_in_doc=0, tokens=None, doc_name=None):
    for term in tokens:
        number_of_words_in_doc += 1
        tf[term][doc_name] += 1
        df[term].append(doc_name)

    documents_words_num[doc_name] = number_of_words_in_doc


def TF_IDF_calculate():
    # Calculate the TF
    for term, doc_tf in tf.items():
        for doc_id, term_freq in doc_tf.items():
            tf[term][doc_id] /= documents_words_num[doc_id]

    # Calculate the TF-IDF scores
    for term, doc_tf in tf.items():
        for doc_id, term_freq in doc_tf.items():
            tfidf[term][doc_id] = term_freq * math.log(documents_count / len(df[term]))


def query_index(query):
    # Tokenize the query
    query_terms = query.lower().split()
    doc_scores = defaultdict(int)

    for term in query_terms:
        if term in tfidf:
            for doc_id, freq in tfidf[term].items():
                doc_scores[doc_id] += freq

    # Sort the documents by their relevance scores
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs

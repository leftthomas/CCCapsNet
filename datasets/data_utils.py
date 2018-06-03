import os
import re
from porterstemmer import Stemmer


stopwords = []
with open(os.path.join('data', 'stopwords.txt'), 'r', encoding='utf-8') as foo:
    for line in foo.readlines():
        line = line.rstrip('\n')
        stopwords.append(line)


def text_preprocess(text):
    # Substitute TAB, NEWLINE and RETURN characters by SPACE.
    text = re.sub('[\t\n\r]', ' ', text)
    # Keep only letters (that is, turn punctuation, numbers, etc. into SPACES).
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Turn all letters to lowercase.
    text = text.lower()
    # Substitute multiple SPACES by a single SPACE.
    text = ' '.join(text.split())
    # Remove words that are less than 3 characters long. For example, removing "he" but keeping "him"
    text = ' '.join(word for word in text.split() if len(word) >= 3)
    # Remove the 524 SMART stopwords (the original stop word list contains 571 words, but there are 47 words contain
    # hyphens, so we removed them, and we found the word 'would' appears twice, so we also removed it, the final stop
    # word list contains 523 words). Some of them had already been removed, because they were shorter than 3 characters.
    # the original stop word list can be found from http://www.lextek.com/manuals/onix/stopwords2.html.
    text = ' '.join(word for word in text.split() if word not in stopwords)
    # Apply Porter's Stemmer to the remaining words.
    stemmer = Stemmer()
    text = ' '.join(stemmer(word) for word in text.split())
    # Substitute multiple SPACES by a single SPACE.
    text = ' '.join(text.split())
    return text

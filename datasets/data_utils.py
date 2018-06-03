import re


def text_preprocess(text):
    # Substitute TAB, NEWLINE and RETURN characters by SPACE.
    text = re.sub('[\t\n\r]', ' ', text)
    # Keep only letters (that is, turn punctuation, numbers, etc. into SPACES).
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Turn all letters to lowercase.
    text = text.lower()
    # Substitute multiple SPACES by a single SPACE.
    text = ' '.join(text.split())
    return text

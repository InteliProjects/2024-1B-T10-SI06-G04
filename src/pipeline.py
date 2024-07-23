import re
import spacy

def extract_links(comment):
    """
    Extract links from a phrase
    Args:
      String: a string

    Returns:
      A string with the phrase
      A string with the links 
    """
    url_regex = r'https?://\S+'
    links = re.findall(url_regex, comment)
    comment = re.sub(url_regex, '', comment)
    return comment.strip()

def tokenize_and_pre_processing(text):
    """
        Tokenize, lemmatize and remove the stop words of a string
        Args:
        A string

        Returns:
        The modified string
    """
    nlp = spacy.load('en_core_web_lg')
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if (token.ent_type_ != '') or (token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'])]
    return lemmas

def to_lower(text):
    """
        Transform uppercase letters from a string into lowercase
        Args:
        A string

        Returns:
        A string
    """
    return text.lower()


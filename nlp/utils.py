import re
import spacy
from nltk.tokenize import word_tokenize  # split sentence to words
from nltk.stem import WordNetLemmatizer  # go to word's root (was -> to be)
from nltk.corpus import stopwords

# load english language model
nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])
wnl = WordNetLemmatizer()


def get_stop_words():
    stop_words = set(stopwords.words('english'))
    with open("/home/selcanyagmuratak/PycharmProjects/Bitirme0/nlp/data/stopwords.txt") as f:
        """
        add desired stop words into global stopwords list.
        """
        for word in f:
            stop_words.add(word.replace('\n', ''))
    return stop_words


# MAYBE NOT USE
def remove_stopwords(tokens):
    stop_words = get_stop_words()
    tokens_list = []
    for word in tokens:
        word = wnl.lemmatize(word.lower())
        if word not in stop_words:
            tokens_list.append(word)
    return list(filter(lambda x: len(x) > 1, tokens_list))


def clean_question(text):
    # regex: Clean non letter characters from the HTML response (syntax).
    cleaned_text = re.sub('[^a-zA-Z]+', ' ', text).strip()
    # token: split text into tokens.
    tokens = word_tokenize(cleaned_text)
    # lemma: Clean stop words like common words(and, you etc.) from the tokens list.
    tokens_lemmatize = remove_stopwords(tokens)
    return tokens_lemmatize


def list_to_string(s):
    string = " "
    return string.join(s)


# return object list in a sentence
def find_objects(text):
    # create spacy
    objects = []
    doc = nlp(text)

    # find nouns in a text
    for token in doc:
        # print('token', token.text, token.pos_, token.dep_, token.ent_type)
        if token.pos_ == 'PRON':
            if token.dep_ == 'nsubj':
                objects.append(token.text)
            elif token.dep_ == 'pobj':
                objects.append(token.text)
        # check token pos
        elif token.pos_ == 'NOUN':
            if token.dep_ == 'attr':
                objects.append(token.text)
            if token.dep_ == 'pobj':
                objects.append(token.text)
            elif token.dep_ == 'nsubj':
                objects.append(token.text)
            elif token.dep_ == 'ROOT':
                objects.append(token.text)
    return objects


if __name__ == '__main__':
    text = 'What color is this bag, please tell me?'
    cl = clean_question(text)
    obj = find_objects(list_to_string(cl))
    print(obj)


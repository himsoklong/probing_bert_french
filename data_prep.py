from conllu import parse_incr, parse
from nltk.tokenize import word_tokenize

def load_conllu(filename):

    """ Parse conllu data 
    Args: filename data in conllu format
    Return : list
    
    """
    with open(filename,"r", encoding="utf-8") as fp:
        data = parse(fp.read())
    sentences = [[token['form'] for token in sentence] for sentence in data]
    tagg = [[token['upos'] for token in sentence] for sentence in data]
    return sentences, tagg


def tokenization(sentence, taggs):

    all_sentence, all_taggs = [], []
    tok_sent = []
    for sent, tagg in zip(sentence, taggs):
        for text in sent:
            tok_sent += word_tokenize(text.lower(),language="french")
        print(tagg, tok_sent)
        assert len(tagg) == len(tok_sent)
        all_sentence.append(tok_sent)
        all_taggs.append(tagg)

    return all_sentence, all_taggs
    
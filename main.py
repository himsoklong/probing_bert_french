import conllu
from data_prep import load_conllu, tokenization
from conllu import parse_incr, parse
import torch
from mangoes.modeling import PretrainedTransformerModelForFeatureExtraction
"""
file = open("./datasets/fr_sequoia-ud-train.conllu", "r", encoding="utf-8")
data = parse(file.read())
i = 0
sentences = [[token['form'] for token in sentence] for sentence in data]
tagg = [[token['upos'] for token in sentence] for sentence in data]
tok_sent = []
for text in sentences[1]:
    if len(text)>1:

        tok_sent += word_tokenize(text.lower(),language="french")
print(tok_sent)
# tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
# camembert = CamembertModel.from_pretrained("camembert-base")

#token = tokenizer.tokenize(" ".join(sentences[1]))
albert_model = PretrainedTransformerModelForFeatureExtraction.load("camembert-base", "camembert-base")
#with torch.no_grad():
#embeddings = albert_model.predict(tok_sent,pre_tokenized=True, to_torch=True)

"""
if __name__ == "__main__":

    filename = "./datasets/fr_sequoia-ud-train.conllu"
    x, y = load_conllu(filename)
    # print(x, y)
    x, y = x[1:2], y[1:2]
    print(x, y)
    print(tokenization(x, y))


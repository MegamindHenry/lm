from json import JSONEncoder
from nltk.tokenize import word_tokenize

class TasaText(object):
    # __slots__ = ['description', 'sents', 'prob_table']

    def __init__(self, descritpion, sents, prob_table=None):
        self.description = descritpion
        self.sents = sents
        if prob_table == None:
            self.prob_table = []
        else:
            self.prob_table = prob_table

    @classmethod
    def from_text(cls, text):
        lines = text.replace('\n', '').split('[S]')

        if lines[0]:
            return cls(lines[0], lines[1:], None)
        return None

    def __str__(self):
        output = self.description
        output += '\n===============\n'
        for sent in self.sents:
            output += '[S] {}\n'.format(sent)
        output += '\n===============\n'
        return output

    def to_sequences(self, context_win=5, open_closed_tag=True):
        text = ' '.join(self.sents)
        sequences = self.construct_sequences(text, context_win)

        return sequences

    def construct_sequences(self, text, context_win):
        tokens = word_tokenize(text)

        length = context_win + 1
        sequences = list()

        # append open tags
        for i in range(context_win):
            open_tag = "<s>"
            # construct open tags
            seq = [open_tag] * (length-i-1) + tokens[:i+1]
            # convert into a line
            line = ' '.join(seq)
            sequences.append(line)

        for i in range(length, len(tokens)+1):
            # select sequence of tokens
            seq = tokens[i - length:i]
            # convert into a line
            line = ' '.join(seq)
            # store
            sequences.append(line)

        # append closed tags 
        closed_tag = "<e>"
        seq = tokens[-length+1:] + [closed_tag]
        line = ' '.join(seq)
        sequences.append(line)

        return sequences


class TasaTextProb(object):
    # __slots__ = ['position', 'gold', 'contexts', 'prob']

    def __init__(self, position, prob):
        self.position = position
        # self.gold = gold
        # self.contexts = contexts
        self.prob = prob


class TasaTextEncoder(JSONEncoder):
    def default(self, object):
        if isinstance(object, TasaTextProb) or isinstance(object, TasaText):
            return object.__dict__
        else:
            # call base class implementation which takes care of
            # raising exceptions for unsupported types
            return JSONEncoder.default(self, object)


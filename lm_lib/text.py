from json import JSONEncoder

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


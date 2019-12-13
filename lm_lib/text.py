class TasaText:
    __slots__ = ['description', 'sents']

    def __init__(self, descritpion, sents):
        self.description = descritpion
        self.sents = sents

    @classmethod
    def from_text(cls, text):
        lines = text.replace('\n', '').split('[S]')

        if lines[0]:
            return cls(lines[0], lines[1:])
        return None

    def __str__(self):
        output = self.description
        output += '\n===============\n'
        for sent in self.sents:
            output += '[S] {}\n'.format(sent)
        output += '\n===============\n'
        return output
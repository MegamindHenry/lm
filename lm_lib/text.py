"""tasa object class file
"""
from json import JSONEncoder
from nltk.tokenize import word_tokenize
import json
import numpy as np

class TasaText(object):

    """TasaText object is a text in a tasa corpus
    
    Attributes:
        description (str): description of tasa text
        name (str): name of the text
        prob_table (list): list of prob table for each position
        segments (list): list of segments in text
        sents (list): list of sentences in text
    """
    
    def __init__(self, name, description, sents, segments, prob_table=None):
        """constructor for TasaText
        
        Args:
            name (str): name
            description (str): description
            sents (list): sents
            segments (list): segments
            prob_table (None, optional): list
        """
        self.name = name
        self.description = description
        self.sents = sents
        self.segments = segments
        if prob_table == None:
            self.prob_table = []
        else:
            self.prob_table = prob_table

    @classmethod
    def from_text(cls, text):
        """construct for a text
        
        Args:
            text (str): tasa text
        
        Returns:
            TasaObject: the TasaText object
        """
        lines = text.replace('\n', '').split('[S]')

        if lines[0]:
            name = lines[0].split()[0]
            segments = word_tokenize(' '.join(lines[1:]))
            return cls(name, lines[0], lines[1:], segments, None)
        return None

    @classmethod
    def from_tt(cls, tt):
        """construct from another TasaText
        
        Args:
            tt (TasaText): tt
        
        Returns:
            TasaText: tt
        """
        return cls(tt.name, tt.description, tt.sents, tt.segments, tt.prob_table)

    def __str__(self):
        """to_string method
        
        Returns:
            str: string
        """
        output = self.description
        output += '\n===============\n'
        for sent in self.sents:
            output += '[S] {}\n'.format(sent)
        output += '\n===============\n'
        return output

    def to_sequences(self, context_win=5, open_closed_tag=True):
        """from TasaText make sequences for text
        
        Args:
            context_win (int, optional): context window length
            open_closed_tag (bool, optional): if need open and closed tag
        
        Returns:
            list: list of content window words
        """
        sequences = self.construct_sequences(context_win)

        return sequences

    def construct_sequences(self, context_win):
        """inner method for construct sequences
        
        Args:
            context_win (int): content window length
        
        Returns:
            list: list of sequences
        """
        tokens = self.segments

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

    def construct_prob_table(self, model, context_win, tokenizer, top_num):
        """make prob table for each position
        
        Args:
            model (keras.model): Trained keras model
            context_win (int): context window length
            tokenizer (keras.tokenizer): Trained tokenizer
            top_num (int): how many top num of words to keep in the table
        """
        lines = self.to_sequences(context_win)
        sequences = tokenizer.texts_to_sequences(lines)
        sequences = np.array(sequences)
        X, y = sequences[:, :-1], sequences[:, -1]

        predicts_raw = model.predict(X)

        prob_table = self.construct_prob(predicts_raw, tokenizer, top_num)

        prob_table_edited = {}
        for i, prob in enumerate(prob_table):
            token = {}
            seq = lines[i].split()
            contexts = ' '.join(seq[:context_win])
            target = seq[-1]
            token['contexts'] = contexts
            token['target'] = target
            token['prob_table'] = prob
            prob_table_edited[i] = token

        self.prob_table = prob_table_edited

    def construct_prob(self, predicts_raw, tokenizer, top_num):
        """make the prob table
        
        Args:
            predicts_raw (np.array): predictions for keras model
            tokenizer (keras.tokenizer): trained tokenizer
            top_num (int): how many top num of words to keep in the table
        
        Returns:
            list: list of prob table for each position
        """
        prob_table_list = []
        for predict in predicts_raw:
            prob_table = dict()

            top_index = np.argsort(predict)[::-1]
            for i in range(top_num):
                index = top_index[i]
                if index == 0:
                    prob_table['<NULL>'] = str(predict[index])
                    continue

                word = tokenizer.index_word[index]
                prob_table[word] = str(predict[index])

            prob_table_list.append(prob_table)

        return prob_table_list


class TasaTextEncoder(JSONEncoder):

    """Encoder for json
    """
    
    def default(self, object):
        """encoder for json
        
        Args:
            object (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        if isinstance(object, TasaText):
            return object.__dict__
        else:
            # call base class implementation which takes care of
            # raising exceptions for unsupported types
            return JSONEncoder.default(self, object)


from collections import UserDict, UserList

from . import utils

class NoteDict(UserDict):
    def __init__(self, word_dict):
        super().__init__(word_dict)

    def __getitem__(self, word):
        if isinstance(word, str):
            key, value = word.split('_')
            if key.startswith("shift"):
                try:
                    return self.data[word]
                except:
                    return float(value)
            else:
                return self.data[word]
        else:
            raise ValueError("word should be int or float")

class NoteList(UserList):
    def __init__(self, words):
        super().__init__(words)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.data[key]
        elif isinstance(key, float):
            return f"shift_{key}"
        else:
            raise ValueError("key should be int or float")

class Vocab:
    def __init__(self, vocab_file, repr):
        # self.i2w = utils.pickle_load(vocab_file)
        # self.w2i = {self.i2w[i]: i for i in range(len(self.i2w))}
        words = utils.pickle_load(vocab_file)

        self.i2f = ['spec', 'bar', 'metric', 'note']
        self.f2i = {}
        for i, f in enumerate(self.i2f):
            self.f2i[f] = i

        self.spec_i2w = ['pad', 'bos', 'eos', 'unk', 'ss', 'se']
        self.spec_w2i = {self.spec_i2w[i]: i for i in range(len(self.spec_i2w))}

        self.bar_i2w = [ word for word in words if word.startswith('bar') ]
        self.bar_w2i = {self.bar_i2w[i]: i for i in range(len(self.bar_i2w))}

        self.metric_i2w = [ word for word in words if 
            word.startswith('metric') or
            word.startswith('beat') or
            word.startswith('tempo') or
            word.startswith('chord')
        ]
        self.metric_w2i = {self.metric_i2w[i]: i for i in range(len(self.metric_i2w))}

        self.note_i2w = NoteList([ word for word in words if
            word.startswith('note') or
            word.startswith('shift') or
            word.startswith('pitch') or
            word.startswith('velocity') or
            word.startswith('duration')
        ])
        self.note_w2i = NoteDict({self.note_i2w[i]: i for i in range(len(self.note_i2w))})

        self.i2w = [self.spec_i2w, self.bar_i2w, self.metric_i2w, self.note_i2w]
        self.w2i = [self.spec_w2i, self.bar_w2i, self.metric_w2i, self.note_w2i]

        for i2w in self.i2w:
            assert i2w[0].endswith('pad')

    def __getitem__(self, key):
        if self.repr == 'remi':
            if isinstance(key, int):
                return self.i2w[key]
            elif isinstance(key, str):
                return self.w2i[key]
            else:
                raise TypeError('key should be int or str')

        elif self.repr == 'cp':
            if isinstance(key[0], str):
                fi = self.f2i[key[0]]
                out = [fi]
                out.extend([self.w2i[fi][w] for w in key[1:]])
            elif isinstance(key[0], int):
                fi = key[0]
                out = [self.i2f[fi]]
                out.extend([self.i2w[fi][i] for i in key[1:]])
            else:
                raise TypeError('key should be int or str')

            return out

        else:
            raise ValueError('repr should be remi or cp')

    def get_family(self, word):
        if word in self.spec_i2w:
            return 'spec'
        elif word.startswith('bar'):
            return 'bar'
        elif word.startswith('beat') or word.startswith('tempo') or word.startswith('chord'):
            return 'metric'
        elif word.startswith('shift') or word.startswith('pitch') or word.startswith('velocity') or word.startswith('duration'):
            return 'note'
        else:
            raise ValueError(f'word "{word}" does not belong to any family')

    def len(self):
        if self.repr == "cp":
            return [len(i2w) for i2w in self.i2w]
        else:
            return len(self.i2w)

    def __repr__(self):
        out = []
        if self.repr == 'remi':
            for word in self.i2w:
                out.append(word)
            return '\n'.join(out)
        elif self.repr == 'cp':
            for f, i in self.f2i.items():
                out.append(f"family: {f}")
                for word in self.i2w[i]:
                    out.append(f"\t{word}")
            return '\n'.join(out)
        else:
            raise ValueError('self.repr should be remi or cp')

if __name__ == '__main__':
    vocab = Vocab('./corpus/ailabs1k7/vocab.pkl', 'remi')
    print(vocab)
    vocab = Vocab('./corpus/ailabs1k7/vocab.pkl', 'cp')
    print(vocab)

    w = vocab[['note', 'shift_0.5', 'pitch_60', 'velocity_100', 'duration_4']]
    print(w)
    print(vocab[w])

    w = vocab[['note', 'shift_4', 'pitch_60', 'velocity_100', 'duration_4']]
    print(w)
    print(vocab[w])


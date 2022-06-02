import json
from argparse import ArgumentParser
from collections import defaultdict

class Trainer:

    def __init__(self, n):
        self.n = n
        self.eos = '<s>'
        self.c_abc = defaultdict(int)
        self.c_abx = defaultdict(int)
        self.u_abx = defaultdict(int)
        self.u_xbc = defaultdict(int)
        self.u_xbx = defaultdict(int)
        self.s_xbx = defaultdict(set)

    def count_ngram(self, ngram):
        assert len(ngram) >= 2

        abc = '|'.join(ngram)
        ab = '|'.join(ngram[:-1])
        bc = '|'.join(ngram[1:])
        b = '|'.join(ngram[1:-1])
        c = ngram[-1]

        self.c_abc[abc] += 1
        self.c_abx[ab] += 1
        if self.c_abc[abc] == 1:
            self.u_abx[ab] += 1
            self.u_xbc[bc] += 1
            self.u_xbx[b] += 1
        self.s_xbx[b] = self.s_xbx[b] | {c}

    def count_sent_ngram(self, n, sent):
        sent = [self.eos] * (n - 1) + sent + [self.eos]
        ngram_iter = zip(*[sent[i:] for i in range(n)])
        for ngram in ngram_iter:
            self.count_ngram(ngram)

    def count_sent(self, sent):
        for n in range(2, self.n + 1):
            self.count_sent_ngram(n, sent)

    def make_vocab(self):
        vocab = [
            (key, value)
            for key, value
            in self.c_abx.items()
            if len(key.split('|')) == 1]
        vocab.sort(key = lambda x: -x[1])
        vocab = [token for token, freq in vocab]
        return vocab

    def to_json(self):
        r_xbx = {key: len(value) for key, value in self.s_xbx.items()}
        dct = {
            'n': self.n,
            'eos': self.eos,
            'c_abc': dict(self.c_abc),
            'c_abx': dict(self.c_abx),
            'u_abx': dict(self.u_abx),
            'u_xbc': dict(self.u_xbc),
            'u_xbx': dict(self.u_xbx),
            'r_xbx': r_xbx,
            'vocab': self.make_vocab()}
        js = json.dumps(dct, indent = 4)
        return js


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--n', type = int, default = 5)
    parser.add_argument('--train', default = 'train.txt')
    parser.add_argument('--lm', default = 'lm.json')
    return parser.parse_args()


def main():
    args = parse_args()

    trainer = Trainer(args.n)

    with open(args.train) as f:
        for line in f:
            sent = line.strip().split()
            trainer.count_sent(sent)

    with open(args.lm, 'w') as f:
        print(trainer.to_json(), file = f)


if __name__ == '__main__':
    main()


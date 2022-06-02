import json
import numpy as np
from argparse import ArgumentParser

class LM:

    def __init__(self, path, d):
        with open(path) as f:
            dct = json.load(f)
        self.n = dct['n']
        self.eos = dct['eos']
        self.c_abc = dct['c_abc']
        self.c_abx = dct['c_abx']
        self.u_abx = dct['u_abx']
        self.u_xbc = dct['u_xbc']
        self.u_xbx = dct['u_xbx']
        self.r_xbx = dct['r_xbx']
        self.vocab = dct['vocab']
        self.d = d

    def predict_lower(self, ngram):
        if len(ngram) == 0:
            return 1 / len(self.vocab)

        abc, ab = '|'.join(ngram), '|'.join(ngram[:-1])

        if (abc in self.u_xbc) and (ab in self.u_xbx):
            alpha = (self.u_xbc[abc] - self.d) / self.u_xbx[ab]
        else:
            alpha = 0

        if ab in self.u_xbx:
            gamma = self.d * self.r_xbx[ab] / self.u_xbx[ab]
        else:
            gamma = 1

        return alpha + gamma * self.predict_lower(ngram[1:])

    def predict(self, ngram):
        abc, ab = '|'.join(ngram), '|'.join(ngram[:-1])

        if (abc in self.c_abc) and (ab in self.c_abx):
            alpha = (self.c_abc[abc] - self.d) / self.c_abx[ab]
        else:
            alpha = 0

        if ab in self.c_abx:
            gamma = self.d * self.u_abx[ab] / self.c_abx[ab]
        else:
            gamma = 1

        return alpha + gamma * self.predict_lower(ngram[1:])


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--lm', default = 'lm.json')
    parser.add_argument('--valid', default = 'valid.txt')
    parser.add_argument('--d', type = float, default = 0.75)
    parser.add_argument('--iters', type = int, default = 20)
    parser.add_argument('--all', action = 'store_true')
    return parser.parse_args()


def sent_to_ngrams(lm, sent):
    sent = ['<s>'] * (lm.n - 1) + sent + ['<s>']
    ngram_iter = zip(*[sent[i:] for i in range(lm.n)])
    return list(ngram_iter)


def calc_ppl(lm, d, data):
    lm.d = d
    probs = [
        -np.log2(lm.predict(ngram))
        for sent in data
        for ngram in sent_to_ngrams(lm, sent)]
    return 2 ** np.mean(probs)


def main():
    args = parse_args()
    lm = LM(args.lm, args.d)

    with open(args.valid) as f:
        data = [line.strip().split() for line in f]

    ppl_list = []
    for d in np.linspace(0, 1, args.iters + 1)[1:]:
        ppl = calc_ppl(lm, d, data)
        ppl_list.append((d, ppl))

    if args.all:
        for d, ppl in ppl_list:
            print('{:.4f}\t{}'.format(d, ppl))
    else:
        d, ppl = min(ppl_list, key = lambda x: x[1])
        print('{:.4f}\t{}'.format(d, ppl))


if __name__ == '__main__':
    main()


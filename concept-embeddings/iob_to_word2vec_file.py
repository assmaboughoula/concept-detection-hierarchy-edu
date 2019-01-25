import word2vec
# https://github.com/danielfrg/word2vec
# https://nbviewer.jupyter.org/github/danielfrg/word2vec/blob/master/examples/word2vec.ipynb

def convert_iob_word2vec(infile_name, outfile_name, sep='\t', ignore_periods=True):
    with open(infile_name) as infile:
        with open(outfile_name, 'w') as outfile:
            cur_phrase = []
            for line in infile:
                word, tag = line.strip().split('\t')
                if ignore_periods and word == '.':
                    continue
                if tag == 'B':
                    outfile.write('_'.join(cur_phrase) + ' ')
                    cur_phrase = [word]
                elif tag == 'I':
                    cur_phrase.append(word)
                elif tag == 'O':
                    outfile.write('_'.join(cur_phrase) + ' ')
                    outfile.write(word + ' ')
            outfile.write('_'.join(cur_phrase) + ' ')
def main():
    convert_iob_word2vec('zhai_tb_iob_tags.txt', 'zhai_tb_phrases.txt')
    word2vec.word2vec('zhai_tb_phrases.txt', 'zhai_tb_embeddings.bin', size=100, verbose=True)
    model = word2vec.load('zhai_tb_embeddings.bin')
    indexes, metrics = model.similar("sentiment_analysis")
    print(model.vocab[indexes])

if __name__ == '__main__':
    main()

import os

if __name__ == '__main__':
    n_sents = 50
    # load CORPORA from environment variable
    corpora_path = os.getenv("CORPORA")
    filepath = os.path.join(corpora_path, 'wi+locness/m2/ABC.train.gold.bea19.m2')
    output_file = os.path.join(corpora_path, 'wi+locness/m2/ABC.train.gold.bea19.n{}.m2'.format(n_sents))

    # read in the data from a file
    with open(filepath, 'r') as f:
        data = f.read()
        # split the data into sentences by empty lines
        sentences = data.split('\n\n')
    
    # save the first N sentences in a new file
    with open(output_file, 'w') as f:
        for sentence in sentences[:n_sents]:
            f.write(sentence + '\n\n')

    # in terminal, use HEAD to subset the data
    # head -n 50 ABC.train.gold.bea19.m2 > ABC.train.gold.bea19.n50.m2
def prepend_slow(infile, outfile, line):
    with open(infile, 'r') as fin:
        with open(outfile, 'w') as fout:
            fout.write(line + "\n")
            for line in fin:
                fout.write(line)

glove_file="glove.840B.300d.txt"
glove_path = "../../glove.840B.300d.txt"
_, tokens, dimensions, _ = glove_file.split('.')
num_lines = 2196017
dims = int(dimensions[:-1])
gensim_file='../../glove_model.txt'
gensim_first_line = "{} {}".format(num_lines, dims)
prepend_slow(glove_path, gensim_file, gensim_first_line)
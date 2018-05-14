
def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)
        
    Yields:
        list of tuples
    """
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch


def read_seq_and_structure(file_name):
    """
    Args:
        file_name: file url. The format of the file must be:
            1. first line: ">" + "file name"
            2. second line: RNA sequence 
            3. secondary structure
            RNA sequence and secondary structure should have the same
            length
    Return:
        seq_structure: list of sequence and structure tuples
            [("GCUUG", ".)).)"), ...]
    """
    seq_structure = []
    file_object = open(file_name, 'r', encoding='utf-8')
    tag = file_object.readline()
    while tag:
        if len(tag) > 0 and tag[0] == '>':
            # read sequence and structure
            sequence = file_object.readline().strip()
            structure  = file_object.readline().strip()
            assert len(sequence) == len(structure)
            seq_structure.append((sequence, structure))
            tag = file_object.readline()
    file_object.close()
    return seq_structure


def encoding_sequence(sequences, seq_dic = {'A': 0, 'C': 1, 'G': 2, 'U': 3}, 
    ss_dic  = {'.': 0, '(': 1, ')': 1}):
    """Encoding RNA sequnce and secondary structure

    Args:
        sequences: list of tuples of (RNA sequence, secondary structure)
    
    Return:
        sequence_structures: list of encoded RNA sequenes, which are list of    
            nucleotide id and pair-status id tuples
            [[(4, 0), (1, 0), (4, 0), ...], [], [], ...]
    """
    sequence_structures = []
    for seq, ss in sequences:
        seq_ss = []
        assert len(seq) == len(ss)
        for n, s in zip(seq, ss):
            n = seq_dic.get(n, 0)
            s = ss_dic.get(s, 0)
            seq_ss.append((n, s))
        sequence_structures.append(seq_ss)

    assert len(sequence_structures) == len(sequences)
    return sequence_structures


def getDataSet(pos_file, neg_file):
    """Read file and create train and dev set

    Args:
        positive_file_url: a path to positive data set. 
            The format of the file must be:
            1. first line: ">" + "file name"
            2. second line: RNA sequence 
            3. secondary structure
            RNA sequence and secondary structure should have the same
            length
        negative_file_url: a path to negative data set
    """
    train = []
    dev = []

    pos_seq_structures = encoding_sequence(read_seq_and_structure(pos_file))
    neg_seq_structures = encoding_sequence(read_seq_and_structure(neg_file))

    # assert len(vec_pos) == len(vec_neg)
    print('Load data from %s successfully, data length: %d'%(pos_file, len(pos_seq_structures)))

    print('Load data from %s successfully, data length: %d'%(neg_file, len(neg_seq_structures)))

    for index in range(len(pos_seq_structures)):
        if index % 5 == 0:
            dev.append((pos_seq_structures[index], 1))
            dev.append((neg_seq_structures[index], 0))
        else:
            train.append((pos_seq_structures[index], 1))
            train.append((neg_seq_structures[index], 0))
        index += 1
    
    return (train, dev)


def pad_sequence(sequences, pad_tok, max_length, window_size=0):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        # create window
        seq_ = create_window(seq_, window_size)
        
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length) - 2*window_size]

    return sequence_padded, sequence_length


def create_window(sequence, window_size):
    """Create window for a giving sequence

    Args:
        sequence: list of data
        window_size: size of the window, int
    """
    windowed_seq = []
    assert len(sequence) > 2*window_size
    # for item in sequence[window_size, -window_size]:
    if window_size == 0:
        return [[i] for i in sequence]
    else:
        for i in range(len(sequence))[window_size: -window_size]:
            neighbors = [i - window_size + j for j in range(2*window_size + 1)]
            windowed_seq.append([sequence[nindex] for nindex in neighbors])
        return windowed_seq

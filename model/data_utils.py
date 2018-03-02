from random import randint

def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (inputs, tags) tuples
        minibatch_size: (int)

    Yields:
        tuple of inputs and labels list

    """
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch

    # x_batch, y_batch = [], []

    # length = len(data)
    # for _ in range(minibatch_size):
    #     rand_index = randint(0, length)

    #     (x, y) = data[rand_index]

    #     x_batch += [x]
    #     y_batch += [y]

    # return x_batch, y_batch

def pad_sequence(inputs, max_length=200, default=[0, 0, 0, 0]):
    """
    Args:
        inputs: list of encoded rna sequence
        max_length: the max_num length of each list

    """
    padded = []
    for item in inputs:
        length = len(item)
        if length > max_length:
            padded.append(item[:max_length])
        else:
            to_pad = max_length - length
            padded.append(item + [default]*to_pad)
    
    return padded

def vectorize(rna_list, array_map, default=[0, 0, 0, 0]):
    """vectorize input rna sequence

    Args:
        rna_list: a list of rna sequences
        array_map: one-hot encoding map
        default: for unknown rna
    
    Returns:
        a list of one-hot encoded rna sequences
    """
    vecRNA = []
    for seq in rna_list:
        vecSeq = []
        for n in seq:
            if array_map.__contains__(n):
                vecSeq.append(array_map[n])
            else:
                vecSeq.append(default)
        vecRNA.append(vecSeq)
    
    return vecRNA

def loadData(filename):
    """load data set and vectorize to array

    Args:
        glove_filename: a path to data set
    
    Returns:
        one-hot encoded rna sequences
    """

    rna_sequences = []
    with open(filename, 'r', encoding='utf-8') as file_object:
        for line in file_object:
            rna_sequences.append(line[:-1])
    print('Data is loaded from: '+filename)

    # vectorize all RNA sequences
    array_map = {
        'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 
        'G': [0, 0, 1, 0], 'U': [0, 0, 0, 1], 
        'N': [0, 0, 0, 0]}
    vec_sequence = vectorize(rna_sequences, array_map)
    assert len(vec_sequence) == len(rna_sequences)

    return vec_sequence

def getDataSet(positive_file_url, negative_file_url):
    """Combine input data with labels

    Args:
        positive_file_url: a path to positive data set
        negative_file_url: a path to negative data set
    
    Returns:
        list of tuple (input, label)
    """

    train = []
    dev = []

    vec_pos = loadData(positive_file_url)
    vec_neg = loadData(negative_file_url)

    # assert len(vec_pos) == len(vec_neg)
    print('Load data from %s successfully, data length: %d'%(positive_file_url, len(vec_pos)))

    print('Load data from %s successfully, data length: %d'%(negative_file_url, len(vec_neg)))

    for index in range(len(vec_pos)):
        if index % 5 == 0:
            dev.append((vec_pos[index], 1))
            dev.append((vec_neg[index], 0))
        else:
            train.append((vec_pos[index], 1))
            train.append((vec_neg[index], 0))
        index += 1
    
    return (train, dev)


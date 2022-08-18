import numpy as np
import tensorflow as tf


"""
Wrapper for loading experiments
"""
def load_experiment(file, permute, pad, orientation):

    assert file in ['mnist.npz', 'fashion_mnist.npz', 'cifar10.npz', 'imdb.npz', 'reuters.npz', 'adding'], 'Dataset requested is not supported.'

    if file == 'mnist.npz':
        (x_train, y_train), (x_test, y_test) = load_mnist(file, permute=permute, pad=pad, orientation=orientation)

    elif file == 'fashion_mnist.npz':
        (x_train, y_train), (x_test, y_test) = load_fashion_mnist(file, permute=permute, pad=pad, orientation=orientation)

    elif file == 'cifar10.npz':
        (x_train, y_train), (x_test, y_test) = load_cifar10(file, permute=permute, pad=pad, orientation=orientation)

    elif file == 'adding':
        (x_train, y_train), (x_test, y_test) = generate_adding_problem(T=750)

    elif file == 'imdb.npz':
        (x_train, y_train), (x_test, y_test) = load_imdb(file, permute=permute, pad=pad, orientation=orientation)

    elif file == 'reuters.npz':
        (x_train, y_train), (x_test, y_test) = load_reuters(file, permute=permute, pad=pad, orientation=orientation)


    n_train = x_train.shape[0]
    n_test = x_test.shape[0]

    if n_train % 32 != 0:
        N = int(np.floor(n_train/32)*32)
        x_train = x_train[:N]
        y_train = y_train[:N]

    if n_test % 32 != 0:
        N = int(np.floor(n_test/32)*32)
        x_test = x_test[:N]
        y_test = y_test[:N]

    return (x_train, y_train), (x_test, y_test)

"""
loads imdb tasks
"""
def load_imdb(file, permute=False, pad=0, orientation=None):
    data = np.load('/grand/rnn-robustness/data/{0}'.format(file))
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    X, Y = np.concatenate((x_train, x_test), axis=0), np.concatenate((y_train, y_test), axis=0)
    X = permute_data(X) if permute else X
    X = X if orientation not in ['post', 'uniform'] else nlp_pad_noise(X, pad=pad, orientation=orientation)

    index = x_train.shape[0]
    (x_train, y_train), (x_test, y_test) = (X[:index], Y[:index]), (X[index:], Y[index:])

    return (x_train, y_train), (x_test, y_test)


"""
loads reuters tasks
"""
def load_reuters(file, permute=False, pad=1000, orientation=None):
    data = np.load('/grand/rnn-robustness/data/{0}'.format(file))
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    X, Y = np.concatenate((x_train, x_test), axis=0), np.concatenate((y_train, y_test), axis=0)
    X = permute_data(X) if permute else X
    X = X if orientation not in ['post', 'uniform'] else nlp_pad_noise(X, pad=pad, orientation=orientation)

    # split according to standard train/test split
    index = x_train.shape[0]
    (x_train, y_train), (x_test, y_test) = (X[:index], Y[:index]), (X[index:], Y[index:])

    return (x_train, y_train), (x_test, y_test)


"""
load fashion mnist tasks
"""
def load_fashion_mnist(file, permute=False, pad=0, orientation=None):

    data = np.load('/grand/rnn-robustness/data/{0}'.format(file))
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    n_train = x_train.shape[0]
    X = tf.concat([x_train, x_test], axis=0)
    X = tf.reshape(X, (X.shape[0],784,1))
    X = tf.cast(X, tf.float32) / 255.
    X = permute_data(X) if permute else X
    X = X if orientation not in ['post', 'uniform'] else add_noise(X, pad=pad, orientation=orientation)

    return (X[:n_train], y_train), (X[n_train:], y_test)


"""
load mnist task
"""
def load_mnist(file, permute=False, pad=0, orientation=None):
    data = np.load('/grand/rnn-robustness/data/{0}'.format(file))
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    n_train = x_train.shape[0]
    X = tf.concat([x_train, x_test], axis=0)
    X = tf.reshape(X, (X.shape[0],784,1))
    X = tf.cast(X, tf.float32) / 255.
    X = permute_data(X) if permute else X
    X = X if orientation not in ['post', 'uniform'] else add_noise(X, pad=pad, orientation=orientation)

    return (X[:n_train], y_train), (X[n_train:], y_test)


"""
load cifar10 tasks
"""
def load_cifar10(file, permute=False, pad=0, orientation=None):
    data = np.load('/grand/rnn-robustness/data/{0}'.format(file))
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    n_train = x_train.shape[0]
    X = tf.concat([x_train, x_test], axis=0)
    X = tf.reshape(X, (X.shape[0],1024,3))
    X = tf.cast(X, tf.float32) / 255.
    X = permute_data(X) if permute else X
    X = X if orientation not in ['post', 'uniform'] else add_noise(X, pad=pad, orientation=orientation)

    return (X[:n_train], y_train), (X[n_train:], y_test)

"""
Generate adding problem dataset/labels
"""
def generate_adding_problem(T=750):
    N = 70000
    features = np.zeros( (N, T, 2) )
    labels = np.zeros( (N) )
    for i in range(0,N):
        rng = np.random.default_rng(i)
        features[i,:,0] = rng.uniform(0,1,T)
        # markers
        marker1 = np.int(np.random.uniform(0,np.floor(T/2)))
        marker2 = np.int(np.random.uniform(np.floor(T/2),T))
        features[i,marker1,1] = 1.0
        features[i,marker2,1] = 1.0
        # labels
        labels[i] = features[i,marker1,0] + features[i,marker2,0]
    # split train and test
    return (features[:60000], labels[:60000]), (features[60000:], labels[60000:])


"""
Appends noise (i.e. padding) specified by pad and orientation
"""
def add_noise(X, pad, orientation=None):
    """
    X: dataset to pad [N samples, time steps, ft dim]
    pad: [# padding steps, orientation] where orientation={'post', 'uniform'}
    """
    (N, T, ft_dim) = X.shape
    Xu = tf.random.uniform(shape=(N,pad,ft_dim), minval=0, maxval=1, seed=0)
    X_out = uniformly_pad(X, Xu) if orientation == 'uniform' else tf.concat([X, Xu], axis=1)

    return X_out


def uniformly_pad(data, noise):

    (N,T,p) = data.shape
    data_single = np.ones(data.shape[1])
    noise_single = np.repeat('a', 1000)

    iters = [data_single, noise_single]
    data_noise = np.stack(list(interleave_evenly(iters)))

    pad_index = np.where(data_noise == 'a')[0]
    data_index = np.where(data_noise != 'a')[0]

    full_data = np.zeros(shape=(N,data_noise.shape[0],p))
    full_data[:, data_index, :] = data
    full_data[:, pad_index, :] = noise

    return full_data



"""
Permutes dataset along axis=1 (temporal index of input)
"""
def permute_data(data):
    p = np.random.RandomState(seed=92916).permutation(data.shape[1])
    return tf.gather(data, p, axis=1)


"""
Padding for IMDB and Reuters tasks (require embedding matrices)
"""
def nlp_pad_noise(data, pad=0, orientation=None):
    N, ft_dim = data.shape
    # fixed: vocabulary size, max input length and padding
    vocab_size = 20000
    max_words = 500

    noise = np.random.RandomState(seed=123).randint(low=4, high=vocab_size, size=(N,pad))

    full_data = np.zeros(shape=(N,1500), dtype='int')

    if orientation == 'post':
        full_data = tf.concat((data, noise), axis=1)

    elif orientation == 'uniform':

        data_single = np.ones(shape=500)
        noise_single = np.repeat('a', 1000)

        iters = [data_single, noise_single]

        m = np.stack(list(interleave_evenly(iters)))

        noise_index = np.where(m == 'a')[0]
        data_index = np.where(m != 'a')[0]
        full_data[:,data_index] = data
        full_data[:,noise_index] = noise

    return full_data


"""
Used for uniform padding -- taken from source code of 'more_itertools'
"""
def interleave_evenly(iterables, lengths=None):
    """
    Interleave multiple iterables so that their elements are evenly distributed
    throughout the output sequence.

    >>> iterables = [1, 2, 3, 4, 5], ['a', 'b']
    >>> list(interleave_evenly(iterables))
    [1, 2, 'a', 3, 4, 'b', 5]

    >>> iterables = [[1, 2, 3], [4, 5], [6, 7, 8]]
    >>> list(interleave_evenly(iterables))
    [1, 6, 4, 2, 7, 3, 8, 5]

    This function requires iterables of known length. Iterables without
    ``__len__()`` can be used by manually specifying lengths with *lengths*:

    >>> from itertools import combinations, repeat
    >>> iterables = [combinations(range(4), 2), ['a', 'b', 'c']]
    >>> lengths = [4 * (4 - 1) // 2, 3]
    >>> list(interleave_evenly(iterables, lengths=lengths))
    [(0, 1), (0, 2), 'a', (0, 3), (1, 2), 'b', (1, 3), (2, 3), 'c']

    Based on Bresenham's algorithm.
    """
    if lengths is None:
        try:
            lengths = [len(it) for it in iterables]
        except TypeError:
            raise ValueError(
                'Iterable lengths could not be determined automatically. '
                'Specify them with the lengths keyword.'
            )
    elif len(iterables) != len(lengths):
        raise ValueError('Mismatching number of iterables and lengths.')

    dims = len(lengths)

    # sort iterables by length, descending
    lengths_permute = sorted(
        range(dims), key=lambda i: lengths[i], reverse=True
    )
    lengths_desc = [lengths[i] for i in lengths_permute]
    iters_desc = [iter(iterables[i]) for i in lengths_permute]

    # the longest iterable is the primary one (Bresenham: the longest
    # distance along an axis)
    delta_primary, deltas_secondary = lengths_desc[0], lengths_desc[1:]
    iter_primary, iters_secondary = iters_desc[0], iters_desc[1:]
    errors = [delta_primary // dims] * len(deltas_secondary)

    to_yield = sum(lengths)
    while to_yield:
        yield next(iter_primary)
        to_yield -= 1
        # update errors for each secondary iterable
        errors = [e - delta for e, delta in zip(errors, deltas_secondary)]

        # those iterables for which the error is negative are yielded
        # ("diagonal step" in Bresenham)
        for i, e in enumerate(errors):
            if e < 0:
                yield next(iters_secondary[i])
                to_yield -= 1
                errors[i] += delta_primary

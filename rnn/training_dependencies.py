import os
import numpy as np
import uuid
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import time
import itertools
import re


def generate_arguments(architectures, learning_rates, hid_dims, epochs, datasets, permutes, pads, orientations):
    """Returns list of  unique user specified treatment combinations"""

    hyper_sets = [architectures, learning_rates, hid_dims, epochs, datasets, permutes, pads, orientations]
    hyper_sets = list(itertools.product(*hyper_sets))

    # remove non-compliant tasks
    configs = []
    for h in hyper_sets:
        pad, orient = h[6], h[7]
        if (pad == 0 and orient != 'None') or (pad != 0 and orient == 'None'):
            continue
        else:
            configs.append(h)

    N = len(configs)

    model_type = np.empty(shape=(N,), dtype='<U13')#
    learning_rate = np.empty(shape=(N,), dtype=float)
    hid_dim = np.empty(shape=(N,), dtype=int)
    epoch = np.empty(shape=(N,), dtype=int)
    dataset = np.empty(shape=(N,), dtype='<U17')# fashion_mnist.npz
    permute = np.empty(shape=(N,), dtype=bool)
    pad = np.empty(shape=(N,), dtype=int)
    orientation = np.empty(shape=(N,), dtype='<U7') # uniform
    identifier = np.empty(shape=(N,), dtype='<U40')

    for i, hypers in enumerate(configs):
        model_type[i] = hypers[0]
        learning_rate[i] = hypers[1]
        hid_dim[i] = hypers[2]
        epoch[i] = hypers[3]
        dataset[i] = hypers[4]
        permute[i] = hypers[5]
        pad[i] = hypers[6]
        orientation[i] = hypers[7]
        identifier[i] = str(uuid.uuid4())


    args_dict = {'model_type': model_type,
                    'learning_rate': learning_rate,
                    'hid_dim': hid_dim,
                    'epochs': epoch,
                    'dataset': dataset,
                    'permute': permute,
                    'pad': pad,
                    'orientation': orientation,
                    'identifier': identifier
                 }

    assert len(architectures) == len(np.unique(args_dict['model_type'])), "Arguments specified incorrectly."
    assert len(learning_rates) == len(np.unique(args_dict['learning_rate'])), "Arguments specified incorrectly."
    assert len(pads) == len(np.unique(args_dict['pad'])), "Arguments specified incorrectly."
    assert len(datasets) == len(np.unique(args_dict['dataset'])), "Arguments specified incorrectly."

    return args_dict

"""
Custom gradient for tf.norm
    - L2 norm
"""
@tf.custom_gradient
def norm(x): #x (bs, hid_dim)
    ϵ = 1.0e-17
    nrm = tf.norm(x, axis=1, keepdims=True)
    def grad(dy):
        return dy * tf.math.divide(x,(nrm + ϵ))
    return nrm, grad

"""
Scaled variance adjoint penalty
"""
@tf.function
def scaled_variance_adjoint_penalty(adjoints):

    tf.stack(adjoints) # (T, bs, dim)
    N = len(adjoints)
    nrms = [norm(adjoints[i]) for i in range(0,N)]

    t1 = tf.zeros(nrms[0].shape)
    for i in range(0, N):
        t1 += nrms[i]

    t1 = t1 / N
    G = tf.zeros(nrms[0].shape)

    for i in range(0, N):
        G += (nrms[i] - t1) ** 2

    return tf.reduce_mean(G)

"""
Complex scaled variance adjoint penalty
"""
@tf.function
def scaled_variance_adjoint_penalty_complex(adjoints):

    tf.stack(adjoints) # (T, bs, dim)
    N = len(adjoints)
    nrms = [norm(adjoints[i]) for i in range(0,N)]

    t1 = tf.zeros(nrms[0].shape, dtype=tf.dtypes.complex64)
    for i in range(0, N):
        t1 += nrms[i]

    t1 = t1 / N
    G = tf.zeros(nrms[0].shape, dtype=tf.dtypes.complex64)

    for i in range(0, N):
        G += (nrms[i] - t1) ** 2

    return tf.reduce_mean(G)

"""
Vanilla RNN class
"""
class RNNRecurrentLayer(tf.keras.layers.Layer):

    def __init__(self, units):
        super().__init__()
        self.units = units
        self.cell = tf.keras.layers.SimpleRNNCell(units)

    def build(self, input_shape):
        self.initial_state = self.cell.get_initial_state(batch_size=input_shape[0], dtype='float32')

    def call(self, inputs):

        input_seq = tf.unstack(inputs, axis=1)
        N = len(input_seq)

        state = self.initial_state
        states = []

        for i in range(0,N):

            state, _ = self.cell(input_seq[i], state)
            states.append(state)

        return states

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'cell': self.cell,
            'initial_state': self.initial_state
        })
        return config

"""
Antisymmetric RNN class
"""
class AntisymmetricLayer(tf.keras.layers.Layer):
    def __init__(self, units, epsilon=0.01, gamma=0.01, sigma=0.01):
        super(AntisymmetricLayer, self).__init__()
        self.units = units
        self.epsilon = epsilon
        self.gamma = gamma
        self.sigma = sigma

    def build(self, input_shape):
        self.V = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer = tf.keras.initializers.RandomNormal(stddev=1/input_shape[-1]),
            trainable=True, name='V'
        )
        self.W = self.add_weight(
            shape=(self.units, self.units),
            initializer = tf.keras.initializers.RandomNormal(stddev=self.sigma/self.units),
            trainable=True, name = 'W'
        )
        self.bias = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True, name='bias')
        self.x0 = self.add_weight(shape=(self.units,), initializer=tf.keras.initializers.Zeros(), trainable=True, name='initial_state')

    def call(self, inputs):


        input_seq = tf.unstack(inputs, axis=1)
        N = len(input_seq)

        state = self.x0 * tf.ones(shape=(tf.shape(inputs)[0], self.units))
        states = []

        M = self.W - tf.transpose(self.W) - self.gamma * tf.eye(self.units)

        for i in range(0,N):

            h = K.dot(input_seq[i], self.V)
            z = h + K.dot(state,M) + self.bias
            tanh_z = tf.keras.activations.tanh(z)
            state = state + self.epsilon * tanh_z
            states.append(state)

        return states

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'sigma': self.sigma
        })
        return config

"""
Lipschitz RNN class
"""
class LipschitzLayer(tf.keras.layers.Layer):

    def __init__(self, units, beta=0.75, gamma_A=0.001, gamma_W=0.001, epsilon=0.03, sigma=0.1/128):
        super(LipschitzLayer, self).__init__()
        self.units = units
        self.beta = beta
        self.gamma_A = gamma_A
        self.gamma_W = gamma_W
        self.epsilon = epsilon
        self.sigma = sigma
        self.M_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.sigma)
        self.U_init = tf.keras.initializers.GlorotUniform()
        self.bias_init = tf.keras.initializers.Zeros()
        self.D_init = tf.keras.initializers.GlorotUniform()

    def build(self, input_shape):
        self.M_A = self.add_weight(shape=(self.units, self.units), initializer=self.M_init, trainable=True, name='M_A')
        self.M_W = self.add_weight(shape=(self.units, self.units), initializer=self.M_init, trainable=True, name='M_W')
        self.U = self.add_weight(shape=(input_shape[-1], self.units), initializer=self.U_init, trainable=True, name='U')
        self.bias = self.add_weight(shape=(self.units, ), initializer=self.bias_init, trainable=True, name='bias')
        self.x0 = self.add_weight(shape=(self.units,), initializer=tf.keras.initializers.Zeros(), trainable=True, name='initial_state')

    def compose_A(self):
        A = (1-self.beta) * (self.M_A + tf.transpose(self.M_A)) + self.beta * (self.M_A - tf.transpose(self.M_A))
        A = A - self.gamma_A * tf.eye(self.units)
        return A

    def compose_W(self):
        W = (1-self.beta) * (self.M_W + tf.transpose(self.M_W)) + self.beta * (self.M_W - tf.transpose(self.M_W))
        W = W - self.gamma_W * tf.eye(self.units)
        return W

    def call(self, inputs):

        input_seq = tf.unstack(inputs, axis=1)
        N = len(input_seq)

        A = self.compose_A()
        W = self.compose_W()

        state = self.x0 * tf.ones(shape=(tf.shape(inputs)[0], self.units))
        states = []

        for i in range(0,N):
            h = K.dot(input_seq[i], self.U)
            z = h + K.dot(state,W) + self.bias
            tanh_z = tf.keras.activations.tanh(z) # σ(z) = σ(Wh+Ux+b)
            Ah = K.dot(state, A)
            state = state + self.epsilon * (Ah + tanh_z) # h + \epsilon * [Ah + σ(Wh+Ux+b)]
            states.append(state)

        return states

    def get_config(self):

        self.beta = beta
        self.gamma_A = gamma_A
        self.gamma_W = gamma_W
        self.epsilon = epsilon
        self.sigma = sigma

        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'beta': self.beta,
            'gamma_A': self.gamma_A,
            'gamma_W': self.gamma_W,
            'epsilon': self.epsilon,
            'sigma': self.sigma
        })
        return config

"""
Modified ReLU activation
"""
def modified_relu(x, bias, ϵ=0.0):

    nrm = K.abs(x)
    M = tf.keras.activations.relu(nrm + bias) / (nrm + ϵ)
    m_relu = tf.cast(M, dtype=tf.dtypes.complex64) * x

    return m_relu

"""
Complex variable uniform initialization
"""
class ComplexUniformInit(tf.keras.initializers.Initializer):

    def __init__(self, minval=-1, maxval=1):
        self.minval = minval
        self.maxval = maxval

    def __call__(self, shape, dtype=tf.dtypes.complex64, **kwargs):
        P = tf.random.uniform(shape=shape, minval=self.minval, maxval=self.maxval)
        return tf.complex(P,P)

    def get_config(self):
        return {'minval': self.minval, 'maxval': self.maxval}

"""
Complex glorot uniform initialization
"""
class ComplexGlorotInit(tf.keras.initializers.Initializer):

    def __init__(self):
        self.glorot = tf.keras.initializers.GlorotUniform()

    def __call__(self, shape, dtype=tf.dtypes.complex64, **kwargs):
        P = self.glorot(shape=shape)
        return tf.cast(P, dtype=tf.dtypes.complex64)

"""
Applies fixed permutation across columns of (d x d) complex matrix
"""

class PermutationMatrix(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(PermutationMatrix, self).__init__()
        self.units = units
        self.permutation = tf.random.shuffle(tf.range(start=0, limit=self.units, dtype=tf.int32))

    def call(self, inputs):
        return tf.gather(inputs, self.permutation, axis=1)

"""
Unitary RNN class
"""
class UnitaryLayer(tf.keras.layers.Layer):

    def __init__(self, units):
        super(UnitaryLayer, self).__init__()
        self.units = units
        self.d_init = tf.keras.initializers.RandomUniform(minval=-np.pi, maxval=np.pi)
        self.r_init = tf.keras.initializers.RandomUniform(minval=-1, maxval=1)
        self.V_init = ComplexGlorotInit()
        self.U_init = tf.keras.initializers.GlorotUniform()
        self.bias_init = tf.keras.initializers.Zeros()
        self.x0_init = ComplexUniformInit(minval=-np.sqrt(3/(2*self.units)), maxval=np.sqrt(3/(2*self.units)))

    def build(self, input_shape):
        self.d1 = self.add_weight(shape=(self.units,), initializer=self.d_init, dtype=tf.dtypes.float32, trainable=True, name='d1')
        self.d2 = self.add_weight(shape=(self.units,), initializer=self.d_init, dtype=tf.dtypes.float32, trainable=True, name='d2')
        self.d3 = self.add_weight(shape=(self.units,), initializer=self.d_init, dtype=tf.dtypes.float32, trainable=True, name='d3')
        self.r1_re = self.add_weight(shape=(self.units,1), initializer=self.r_init, dtype=tf.dtypes.float32, trainable=True, name='r1_re')
        self.r1_im = self.add_weight(shape=(self.units,1), initializer=self.r_init, dtype=tf.dtypes.float32, trainable=True, name='r1_im')
        self.r2_re = self.add_weight(shape=(self.units,1), initializer=self.r_init, dtype=tf.dtypes.float32, trainable=True, name='r2_re')
        self.r2_im = self.add_weight(shape=(self.units,1), initializer=self.r_init, dtype=tf.dtypes.float32, trainable=True, name='r2_im')

        self.Pi = PermutationMatrix(units=self.units)
        self.V = self.add_weight(shape=(input_shape[-1], self.units), initializer=self.V_init, dtype=tf.dtypes.complex64, trainable=True, name='V')
        #self.U = self.add_weight(shape=(2*self.units, self.output_dim), initializer='glorot_uniform', trainable=True, name='U')
        self.bias = self.add_weight(shape=(self.units,), initializer=self.bias_init, dtype=tf.dtypes.float32, trainable=True, name='bias')
        #self.bias_out = self.add_weight(shape=(self.output_dim,), initializer=self.bias_init, trainable=True, name='bias_out')
        self.activation = modified_relu
        self.x0 = self.add_weight(shape=(self.units,), initializer=self.x0_init, dtype=tf.dtypes.complex64, trainable=False, name='initial_state')

    def compose_D(self):
        D1 = tf.linalg.diag(tf.complex(tf.cos(self.d1), tf.sin(self.d1)))
        D2 = tf.linalg.diag(tf.complex(tf.cos(self.d2), tf.sin(self.d2)))
        D3 = tf.linalg.diag(tf.complex(tf.cos(self.d3), tf.sin(self.d3)))
        return (D1, D2, D3)

    def compose_R(self):

        v1 = tf.complex(self.r1_re, self.r1_im)
        v1_star = tf.math.conj(v1)
        v2 = tf.complex(self.r2_re, self.r2_im)
        v2_star = tf.math.conj(v2)

        t1 = tf.matmul(v1,tf.transpose(v1_star)) / tf.linalg.norm(v1)**2
        t2 = tf.matmul(v2,tf.transpose(v2_star)) / tf.linalg.norm(v2)**2

        ident = tf.eye(self.r1_re.shape[0], dtype=tf.dtypes.complex64)

        R1 = ident - 2 * t1
        R2 = ident - 2 * t2

        return (R1, R2)

    def compose_W(self):
        # W = D3 R2 F^-1 D2 Π R1 F D1
        (D1, D2, D3) = self.compose_D()
        (R1, R2) = self.compose_R()

        t1 = K.dot(R1, tf.signal.fft(D1))
        t2 = self.Pi(t1)
        t3 = tf.signal.ifft(K.dot(D2, t2))
        t4 = K.dot(R2, t3)
        W = K.dot(D3, t4)

        return W

    def call(self, inputs):

        input_seq = tf.unstack(inputs, axis=1)
        N = len(input_seq)

        W = self.compose_W()

        state = self.x0 * tf.ones(shape=(tf.shape(inputs)[0], self.units), dtype=tf.dtypes.complex64)
        states = []

        for i in range(0,N):

            h = K.dot(input_seq[i], self.V)
            output = h + K.dot(state, W)
            state = self.activation(output, self.bias)
            states.append(state)

        return states

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'units': self.units,
        })
        return config


"""
Activation acting on time scale parameter (i.e. c_i)
"""
def mod_tanh(inputs):
    return 0.5 + 0.5 * tf.keras.activations.tanh(inputs/2)

"""
UniCORNN Layer -- assumes fixed layer depth=2)
"""
class UnICORNNLayer(tf.keras.layers.Layer):

    def __init__(self, units, ft_dim, epsilon=0.03, alpha=0.9, L=2):
        super(UnICORNNLayer, self).__init__()
        self.units = units
        self.ft_dim = ft_dim
        self.w_init = tf.keras.initializers.RandomUniform(minval=0, maxval=1)
        self.V_init = tf.keras.initializers.HeUniform()
        self.bias_init = tf.keras.initializers.Zeros()
        self.c_init = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
        #self.D_init = tf.keras.initializers.GlorotUniform()
        self.epsilon = epsilon
        self.alpha = alpha
        self.state_init = tf.keras.initializers.Zeros()
        self.L = L
        self.rec_activation = tf.keras.activations.tanh
        self.time_activation = mod_tanh
        self.w1 = self.add_weight(shape=(self.units,), initializer=self.w_init, trainable=True, name='w1')
        self.w2 = self.add_weight(shape=(self.units,), initializer=self.w_init, trainable=True, name='w2')
        self.V1 = self.add_weight(shape=(self.ft_dim, self.units), initializer=self.V_init, trainable=True, name='V1')
        self.V2 = self.add_weight(shape=(self.units, self.units), initializer=self.V_init, trainable=True, name='V2')
        self.b1 = self.add_weight(shape=(self.units, ), initializer=self.bias_init, trainable=True, name='b1')
        self.b2 = self.add_weight(shape=(self.units, ), initializer=self.bias_init, trainable=True, name='b2')
        self.c1 = self.add_weight(shape=(self.units, ), initializer=self.c_init, trainable=True, name='c1')
        self.c2 = self.add_weight(shape=(self.units, ), initializer=self.c_init, trainable=True, name='c2')

    def call(self, inputs):

        input_seq = tf.unstack(inputs, axis=1)
        N = len(input_seq)

        y1 = tf.zeros(shape=(tf.shape(inputs)[0], self.units))
        z1 = tf.zeros(shape=(tf.shape(inputs)[0], self.units))
        y2 = tf.zeros(shape=(tf.shape(inputs)[0], self.units))
        z2 = tf.zeros(shape=(tf.shape(inputs)[0], self.units))

        states = []

        for i in range(0,N):
            # 1. compute layer 1 z1
            # 2. compute layer 1 y1
            # 3. compute layer 2 z1
            # 4. compute layer 2 y2

            # layer 1: z1_nxt, y1_nxt
            δ1 = self.epsilon * self.time_activation(self.c1)
            h = K.dot(input_seq[i], self.V1)
            h = self.rec_activation(h + tf.multiply(self.w1, y1) + self.b1)
            z1_nxt = z1 - δ1 * (h + self.alpha * y1)
            y1_nxt = y1 + δ1 * z1_nxt

            # layer 2: z2_nxt, y2_nxt
            δ2 = self.epsilon * self.time_activation(self.c2)
            h = K.dot(y1_nxt, self.V2)
            h = self.rec_activation(h + tf.multiply(self.w2, y2) + self.b2)
            z2_nxt = z2 - δ2 * (h + self.alpha * y2)
            y2_nxt = y2 + δ2 * z2_nxt

            # store states
            state_i = tf.concat([y1_nxt, z1_nxt, y2_nxt, z2_nxt], axis=1)
            states.append(state_i)

            # reset states
            y1 = state_i[:, :self.units] # y1_nxt
            z1 = state_i[:, self.units:2*self.units] # z1_nxt
            y2 = state_i[:, 2*self.units:3*self.units] # y2_nxt
            z2 = state_i[:, 3*self.units:] # z2_nxt

        return states


"""
Description: LSTM layer class
"""
class LSTMLayer(tf.keras.layers.Layer):
    def __init__(self, hid_dim):
        super().__init__()
        self.cell = tf.keras.layers.LSTMCell(hid_dim)
        self.hid_dim = hid_dim

    def build(self, input_shape):
        initial_states = self.cell.get_initial_state(batch_size=input_shape[0], dtype='float32')
        self.x0 = initial_states[0]
        self.c0 = initial_states[1]

    def call(self, inputs):

        input_seq = tf.unstack(inputs, axis=1)
        N = len(input_seq)

        state = self.x0
        cell_state = self.c0

        states = []
        cell_states = []

        for i in range(0,N):
            _, [state, cell_state] = self.cell(input_seq[i], [state, cell_state])
            states.append(state)
            cell_states.append(cell_state)

        return states #, cell_states

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'cell': self.cell,
            'hid_dim': self.hid_dim,
            #'x0': self.x0,
            #'c0': self.c0
        })
        return config


"""
Description: GRU layer class
"""
class GRULayer(tf.keras.layers.Layer):
    def __init__(self, hid_dim):
        super().__init__()
        self.cell = tf.keras.layers.GRUCell(hid_dim)
        self.hid_dim = hid_dim

    def build(self, input_shape):
        self.x0 = self.cell.get_initial_state(batch_size=input_shape[0], dtype='float32')

    def call(self, inputs):

        input_seq = tf.unstack(inputs, axis=1)
        N = len(input_seq)

        state = self.x0
        states = []

        for i in range(0,N):
            _, state = self.cell(input_seq[i], state)
            states.append(state)

        return states

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            #'cell': self.cell,
            'hid_dim': self.hid_dim,
            #'x0': self.x0
        })
        return config


def differential(f, A, E):
    """ Computes the differential of f at A when acting on E:  (df)_A(E) """
    n = A.shape[0]
    Z = tf.zeros((n,n))

    top = tf.concat([A, E], axis=1)
    bottom = tf.concat([Z, A], axis=1)
    M = tf.concat([top, bottom], axis=0)

    return f(M)[:n, n:]

def update_expA(model, grad_B, lr):

    η = lr * 0.1
    B = model.get_layer('exponential_rnn_layer').B
    A = model.get_layer('exponential_rnn_layer').A
    E = 0.5 * (tf.matmul(tf.transpose(grad_B), B) - tf.matmul(tf.transpose(B), grad_B))
    grad_A = tf.matmul(B, differential(tf.linalg.expm, tf.transpose(A), E))
    update = A + η * grad_A
    model.get_layer('exponential_rnn_layer').A.assign(update)

    return

def create_diag_(A, diag):
    n = A.shape[0]
    diag_z = np.zeros(n-1)
    diag_z[::2] = diag
    A_init = tf.linalg.diag(diag_z, k=1)
    A_init = A_init - tf.transpose(A_init)
    return A_init

def cayley_init_(A):
    size = A.shape[0] // 2
    diag = tf.random.uniform(shape=(size,), minval=0., maxval=np.pi / 2.)
    diag = -tf.sqrt( (1. - tf.cos(diag)) / (1. + tf.cos(diag)) )

    return create_diag_(A, diag)

class cayleyInit(tf.keras.initializers.Initializer):

    def __init__(self, mean=0, stddev=1):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, shape, dtype=None, **kwargs):
        A = tf.random.normal(shape, mean=self.mean, stddev=self.stddev, dtype=dtype)
        A_cayley = cayley_init_(A)
        return tf.cast(A_cayley, tf.float32)

    def get_config(self):  # To support serialization
        return {"mean": self.mean, "stddev": self.stddev}


class modrelu(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(modrelu, self).__init__()
        self.dim = dim

    def build(self, inputs):
        self.bias = tf.Variable(tf.random.uniform(shape=(self.dim,), minval=-0.01, maxval=0.01), trainable=True, name='bias')

    def call(self, inputs):
        nrm = tf.abs(inputs)
        biased_nrm = nrm + self.bias
        magnitude = tf.keras.activations.relu(biased_nrm)
        phase = tf.sign(inputs)
        return phase * magnitude

    def get_config(self):
        base_config = super(modrelu, self).get_config()
        config = {'bias': self.bias}
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

"""
Exponential RNN class
"""
class ExponentialRNNLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(ExponentialRNNLayer, self).__init__()
        self.units = units

    def build(self, input_shape):

        self.T = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=1/input_shape[-1]),
            trainable=True,
            name='T'
        )

        self.A = self.add_weight(
            shape=(self.units, self.units),
            initializer = cayleyInit,
            trainable=False,
            name='A'
        )

        self.B = tf.Variable(tf.linalg.expm(self.A), trainable=True)

        self.activation = modrelu(self.units)

        self.h0 = self.add_weight(shape=(self.units,), initializer=tf.keras.initializers.Zeros(), trainable=True, name='initial_state')


    def reset_parameters(self):
        # retraction to tangent space
        A = tf.linalg.band_part(self.A, 0, -1) # upper triangular matrix
        A = A - tf.transpose(A)
        self.A.assign(A)
        # assign B from retraction
        self.B.assign(tf.linalg.expm(self.A))


    def call(self, inputs): #[batch, T, p]

        input_seq = tf.unstack(inputs, axis=1)
        N = len(input_seq)

        state = tf.ones(shape=(tf.shape(inputs)[0], self.units)) * self.h0
        states = []

        for i in range(0,N):

            h = K.dot(input_seq[i], self.T)
            h = h + K.dot(state, self.B)
            h = self.activation(h)
            state = h
            states.append(state)

        return states

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'units': self.units,
        })
        return config

"""
Coadjoint Class Implementation
"""
class CoadjointModel(tf.keras.Model):

    def compile(self, model_name, optimizer, loss_fn, adj_penalty, lc_weights, embed=False):
        self.model_name = model_name
        self.embed = embed
        self.loss_fn = loss_fn
        self.adj_penalty = adj_penalty
        self.lc_weights = lc_weights
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.penalty_tracker = tf.keras.metrics.Mean(name='penalty')
        self.first_adjoint = tf.keras.metrics.Mean(name='first-adjoint')
        self.last_adjoint = tf.keras.metrics.Mean(name='last-adjoint')
        if self.loss_fn.name in ['binary_crossentropy', 'mean_squared_error']:
            self.accuracy_tracker = tf.keras.metrics.BinaryAccuracy(name='accuracy')
        else:
            self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

        super(CoadjointModel, self).compile(optimizer=optimizer)

    @tf.function
    def train_step(self, data):

        # compute B=e**A prior to each batch update
        if self.model_name == 'exponential':
            self.get_layer('exponential_rnn_layer').reset_parameters()

        #self.layers[2].reset_parameters() if self.embed else self.layers[1].reset_parameters()

        x, y = data

        with tf.GradientTape() as t2:
            with tf.GradientTape(persistent=True) as t1:
                outputs = self(x, training=True)
                L = self.loss_fn(y, outputs[0])

            dL_dW = t1.gradient(L, self.trainable_variables)
            dL_dX = t1.gradient(L, outputs[1])

            G = self.adj_penalty( dL_dX )

        dG_dW = t2.gradient(G, self.trainable_variables)

        del t1

        # for embedding layer compatibility
        if self.embed:
            dL_dW[0] = tf.convert_to_tensor(dL_dW[0])
            dG_dW[0] = tf.convert_to_tensor(dG_dW[0])

        dL_plus_G_dW = [
            tf.add( x[0] * self.lc_weights[0], x[1] * self.lc_weights[1] )
            for x in zip(dL_dW, dG_dW)
            ]

        # update parameters
        self.optimizer.apply_gradients(zip(dL_plus_G_dW, self.trainable_variables))


        if self.model_name == 'exponential':
            # apply gradient update to A
            if self.embed:
                update_expA(self, dL_plus_G_dW[2], self.optimizer.learning_rate) # Prop 4.1 gradient
            else:
                update_expA(self, dL_plus_G_dW[1], self.optimizer.learning_rate) # Prop 4.1 gradient

            self.get_layer('exponential_rnn_layer').reset_parameters()


        λ1 = tf.reduce_mean(tf.norm(dL_dX[0]))
        λN = tf.reduce_mean(tf.norm(dL_dX[-1]))

        #Update Metrics
        self.loss_tracker.update_state(L)
        self.penalty_tracker.update_state(G)
        self.accuracy_tracker.update_state(y,outputs[0])
        self.first_adjoint.update_state(λ1)
        self.last_adjoint.update_state(λN)

        return {"loss": self.loss_tracker.result(),
                "penalty": self.penalty_tracker.result(),
                "accuracy": self.accuracy_tracker.result(),
                "λ1": self.first_adjoint.result(),
                "λN": self.last_adjoint.result()
                }

    @property
    def metrics(self):
        return [self.loss_tracker, self.penalty_tracker, self.accuracy_tracker,
                    self.first_adjoint, self.last_adjoint]

    @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        with tf.GradientTape() as t1:
            outputs = self(x)
            L = self.loss_fn(y, outputs[0]) # (true label, prediction)

        #Compute Adjoints
        dL_dX = t1.gradient(L, outputs[1])

        #Compute Adjoint Penalty
        G = self.adj_penalty( dL_dX )

        #Update Metrics
        λ1 = tf.reduce_mean(norm(dL_dX[0]))
        λN = tf.reduce_mean(norm(dL_dX[-1]))
        self.loss_tracker.update_state(L)
        self.penalty_tracker.update_state(G)
        self.accuracy_tracker.update_state(y,outputs[0])
        self.first_adjoint.update_state(λ1)
        self.last_adjoint.update_state(λN)

        return {"loss": self.loss_tracker.result(),
                "penalty": self.penalty_tracker.result(),
                "accuracy": self.accuracy_tracker.result(),
                "λ1": self.first_adjoint.result(),
                "λN": self.last_adjoint.result()
                }

"""
Creates coadjoint model from provided arguments
"""
def make_model(name, T, ft_dim, hid_dim, out_dim, penalty_weight, learning_rate, embed=False):

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if out_dim > 1:
        output_activation = 'softmax'
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
    else:
        output_activation = 'sigmoid'
        loss = tf.keras.losses.BinaryCrossentropy()

    layers = {'rnn': RNNRecurrentLayer(hid_dim),
              'lstm': LSTMLayer(hid_dim),
              'gru': GRULayer(hid_dim),
              'antisymmetric': AntisymmetricLayer(hid_dim, epsilon=0.01, gamma=0.01, sigma=0.01),
              'lipschitz': LipschitzLayer(hid_dim, beta=0.75, gamma_A=0.001, gamma_W=0.001, epsilon=0.03, sigma=0.1/128),
              'unitary': UnitaryLayer(hid_dim),
              'exponential': ExponentialRNNLayer(hid_dim),
              'unicornn': UnICORNNLayer(hid_dim, ft_dim, epsilon=0.03, alpha=0.9, L=2)
             }

    if embed:
        if name == 'unitary':
            penalty = scaled_variance_adjoint_penalty_complex
            inputs = tf.keras.Input( shape=(T,), name='input_layer', dtype=tf.dtypes.complex64)
            embed = tf.keras.layers.Embedding(20000, ft_dim, input_length=T)
            rec_layer = layers[name]
            dense_layer = tf.keras.layers.Dense(out_dim, activation=output_activation, name='output-layer')
            vec_inputs = embed(inputs)
            states = rec_layer(vec_inputs)
            terminal_state = states[-1]
            terminal_state = tf.concat([tf.math.real(terminal_state), tf.math.imag(terminal_state)], axis=1)
            outputs = dense_layer(terminal_state)
        else:
            penalty = scaled_variance_adjoint_penalty
            inputs = tf.keras.Input( shape=(T,), batch_size=32, name='input-layer')
            embed = tf.keras.layers.Embedding(20000, ft_dim, input_length=T)
            rec_layer = layers[name]
            dense_layer = tf.keras.layers.Dense(out_dim, activation=output_activation, name='output-layer')
            vec_inputs = embed(inputs)
            states = rec_layer(vec_inputs)
            outputs = dense_layer(states[-1])

    else:
        if name == 'unitary':
            penalty = scaled_variance_adjoint_penalty_complex
            inputs = tf.keras.Input( shape=(T,ft_dim), batch_size=32, name='input-layer', dtype=tf.dtypes.complex64)
            rec_layer = layers[name]
            dense_layer = tf.keras.layers.Dense(out_dim, activation=output_activation, name='output-layer')
            states = rec_layer(inputs)
            terminal_state = states[-1]
            terminal_state = tf.concat([tf.math.real(terminal_state), tf.math.imag(terminal_state)], axis=1)
            outputs = dense_layer(terminal_state)

        else:
            penalty = scaled_variance_adjoint_penalty
            inputs = tf.keras.Input( shape=(T,ft_dim), batch_size=32, name='input-layer')
            rec_layer = layers[name]
            dense_layer = tf.keras.layers.Dense(out_dim, activation=output_activation, name='output-layer')
            states = rec_layer(inputs)
            outputs = dense_layer(states[-1])

    model = CoadjointModel(
            inputs=inputs,
            outputs=[outputs, states]
    )

    # Compile Model
    model.compile(optimizer=optimizer,
                loss_fn=loss,
                lc_weights=[1.0, penalty_weight],
                adj_penalty=penalty,
                model_name=name,
                embed=embed
                )
    model.run_eagerly = True
    return model


"""
Callback: Collect Epoch Training Times
"""
class TimeCallback(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class CheckpointModel(tf.keras.callbacks.Callback):
    """
    Description:
        Callback that saves training arguments, training history and
        model/optimizer weights when subsequent epoch will exceed maximum
        ensemble execution time.
    """
    def __init__(self, training_args, pid):
        self.ensemble_end_time = training_args.ensemble_end_time
        self.checkpoint_path = f'/grand/rnn-robustness/test-checkpoints/model-{training_args.identifier}-{pid}'
        self.history_name = f'training-history-{training_args.identifier}-{pid}'
        self.epoch_times = []
        self.avg_epoch_time = -float('inf')
        self.training_args = training_args

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

        time_remaining = self.ensemble_end_time - time.time()

        if time_remaining < self.avg_epoch_time + 600: # 10 minute buffer
            # create checkpoint directory
            dir_flag = os.path.exists(self.checkpoint_path)

            if not dir_flag:
                os.makedirs(self.checkpoint_path)

            # save model and optimizer weights
            self.model.save_weights(f'{self.checkpoint_path}/checkpoint-weights')
            history = self.model.history.history
            np.savez(f'{self.checkpoint_path}/{self.history_name}.npz', history=history, training_args=self.training_args, epoch_times=self.epoch_times)

    def on_epoch_end(self, batch, logs={}):
        epoch_end_time = time.time()
        self.epoch_times.append(epoch_end_time - self.epoch_time_start)
        self.avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)

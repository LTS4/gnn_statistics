import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import numpy as np


def select_features_logreg_l1(features, targets, num_features):
    if num_features is None:
        return features
    labels = np.argmax(targets, axis=1)
    clf = LogisticRegression(penalty='l1', C=1e1, fit_intercept=False, solver='liblinear', multi_class='ovr')
    clf.fit(features, labels)
    model = SelectFromModel(clf, prefit=True, max_features=num_features, threshold=-np.inf)
    return model.transform(features)


def select_random_features(features, num_features):
    old_num_features = features.shape[1]
    random_scrumbler = 1 / np.sqrt(old_num_features) * np.random.randn(old_num_features, num_features).astype(np.float32)
    features = features @ random_scrumbler
    return features


def select_features(features, num_features, targets, random_selection):
    if num_features < features.shape[1]:
        if random_selection:
            features = select_random_features(features, num_features)

        else:
            features = select_features_logreg_l1(features, targets, num_features)
    return features


def init_w_glorot(shape, name=None):
    """ Initialize a tensor of given shape according to Glorot initialization. """
    print("Initialize", shape, name)
    w = tf.get_variable(name, shape, dtype=tf.float32,
                        initializer=tf.glorot_uniform_initializer(), trainable=False)
    return w


def graph_convolution(inputs, sparse_renormalized_laplacian, weights, input_is_sparse=False):
    """Implements the graph convolution operation Â * inputs * weights, where
    Â is the renormalized Laplacian Â = D~^-0.5 * A~ * D~^-0.5 with
    A~ = A + I_N (adjacency matrix with added self-loops) and
    D~ = diagonal matrix of node degrees deduced from A~.
    """
    if input_is_sparse:
        output = tf.sparse_tensor_dense_matmul(inputs, weights)
    else:
        output = tf.matmul(inputs, weights)
    return tf.sparse_tensor_dense_matmul(sparse_renormalized_laplacian, output)


def compute_adj_polynomials(graph_adj, p_max):
    """Compute a list of powers of graph_adj up to p_max"""
    _p = p_max + 1
    n_nodes = int(graph_adj.get_shape()[0])

    if type(graph_adj) == tf.SparseTensor:
        polynomials = list(tf.sparse.eye(n_nodes, n_nodes))

        # Not sure it works
        for i in range(1, _p):
            polynomials.append(tf.sparse_tensor_dense_matmul(graph_adj, polynomials[-1]))
    else:
        polynomials = list(tf.eye(n_nodes))
        for i in range(1, _p):
            polynomials.append(tf.matmul(graph_adj, polynomials[-1]))

    return polynomials


def graph_polynomial_convolution_layer(graph_adj, _p_max, output_dim, inputs,
                                       input_is_sparse, name):
    """ f(X) = sum_{i < _p_max} c_i A^i X """
    input_dim = int(inputs.get_shape()[1])

    with tf.name_scope(name):
        coefs = init_w_glorot([_p_max, input_dim, output_dim], "weights-" + name)

        for i in range(_p_max):
            matmul = tf.sparse_tensor_dense_matmul if input_is_sparse else tf.matmul
            x = matmul(inputs, coefs[i])

            for j in range(i):
                x = tf.sparse_tensor_dense_matmul(graph_adj, x)

            output = x if i == 0 else output + x
    return output

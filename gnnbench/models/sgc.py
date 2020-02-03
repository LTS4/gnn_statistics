import tensorflow as tf
import tensorflow.contrib.slim as slim
from sacred import Ingredient

from gnnbench.data.preprocess import row_normalize, renormalize_adj
from gnnbench.models.base_model import GNNModel
from gnnbench.models.util import select_features
from gnnbench.util import dropout_supporting_sparse_tensors, to_sparse_tensor


def fully_connected(output_dim, inputs, activation_fn, dropout_prob, weight_decay,
                    name, input_is_sparse=False):
    with tf.name_scope(name):
        input_dim = int(inputs.get_shape()[1])
        weights = tf.get_variable("%s-weights" % name, [input_dim, output_dim], dtype=tf.float32,
                                  initializer=tf.glorot_uniform_initializer(),
                                  regularizer=slim.l2_regularizer(weight_decay))
        bias = tf.get_variable("%s-bias" % name, [output_dim], dtype=tf.float32,
                               initializer=tf.zeros_initializer())

        # Apply dropout to inputs if required
        inputs = dropout_supporting_sparse_tensors(inputs, 1 - dropout_prob)

        if input_is_sparse:
            output = tf.sparse_tensor_dense_matmul(inputs, weights)
        else:
            output = tf.matmul(inputs, weights)
        output += bias
        if activation_fn is not None:
            output = activation_fn(output)
        return output


class SimpleGraphConvolution(GNNModel):
    def __init__(self, features, graph_adj, targets, nodes_to_consider, num_layers,
                 weight_decay, normalize_features, num_features, random_selection):
        self.num_layers = num_layers
        self.normalize_features = normalize_features
        self.nodes_to_consider = nodes_to_consider
        self.num_features = num_features
        self.random_selection = random_selection
        features = select_features(features, num_features, targets, random_selection)
        with tf.name_scope('extract_relevant_nodes'):
            targets = tf.gather(targets, nodes_to_consider)
        super().__init__(features, graph_adj, targets)
        self.weight_decay = weight_decay
        self.num_features = num_features
        self.random_selection = random_selection

        self._build_model_graphs()

    def _inference(self):
        with tf.name_scope('inference'):
            x = self.features
            output = fully_connected(output_dim=self.targets.shape[1],
                                     inputs=x,
                                     activation_fn=None,
                                     dropout_prob=0,
                                     weight_decay=self.weight_decay,
                                     name="gc0",
                                     input_is_sparse=True)
        with tf.name_scope('extract_relevant_nodes'):
            return tf.gather(output, self.nodes_to_consider)

    def _preprocess_features(self, features):
        """ The propagation step is made here"""
        if self.normalize_features:
            features = row_normalize(features)
        for i in range(self.num_layers):
            features = self.graph_adj @ features
        return to_sparse_tensor(features)

    def _preprocess_adj(self, graph_adj):
        return renormalize_adj(graph_adj)
        # return to_sparse_tensor(renormalize_adj(graph_adj))


MODEL_INGREDIENT = Ingredient('model')


@MODEL_INGREDIENT.capture
def build_model(graph_adj, node_features, labels, dataset_indices_placeholder,
                train_feed, trainval_feed, val_feed, test_feed,
                weight_decay, normalize_features,
                num_layers, hidden_size, dropout_prob, num_features, random_selection):
    dropout = tf.placeholder(dtype=tf.float32, shape=[])
    train_feed[dropout] = dropout_prob
    trainval_feed[dropout] = False
    val_feed[dropout] = False
    test_feed[dropout] = False

    return SimpleGraphConvolution(node_features, graph_adj, labels, dataset_indices_placeholder,
                                  num_layers, weight_decay, normalize_features, num_features, random_selection)

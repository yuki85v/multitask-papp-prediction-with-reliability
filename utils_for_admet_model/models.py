import torch
from torch import nn
from dgllife.model.gnn.attentivefp import AttentiveFPGNN
from dgllife.model.readout.attentivefp_readout import AttentiveFPReadout
from dgllife.model.gnn import GCN
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax


class GraphEncoderGCN(nn.Module):
    """
    GCN from `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`__

    Parameters
    ----------
    in_feats : int
        Number of input node features.
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the size of node representations after the i-th GCN layer.
        ``len(hidden_feats)`` equals the number of GCN layers.  By default, we use
        ``[64, 64]``.
    gnn_norm : list of str
        ``gnn_norm[i]`` gives the message passing normalizer for the i-th GCN layer, which
        can be `'right'`, `'both'` or `'none'`. The `'right'` normalizer divides the aggregated
        messages by each node's in-degree. The `'both'` normalizer corresponds to the symmetric
        adjacency normalization in the original GCN paper. The `'none'` normalizer simply sums
        the messages. ``len(gnn_norm)`` equals the number of GCN layers. By default, we use
        ``['none', 'none']``.
    activation : list of activation functions or None
        If not None, ``activation[i]`` gives the activation function to be used for
        the i-th GCN layer. ``len(activation)`` equals the number of GCN layers.
        By default, ReLU is applied for all GCN layers.
    residual : list of bool
        ``residual[i]`` decides if residual connection is to be used for the i-th GCN layer.
        ``len(residual)`` equals the number of GCN layers. By default, residual connection
        is performed for each GCN layer.
    batchnorm : list of bool
        ``batchnorm[i]`` decides if batch normalization is to be applied on the output of
        the i-th GCN layer. ``len(batchnorm)`` equals the number of GCN layers. By default,
        batch normalization is applied for all GCN layers.
    dropout : list of float
        ``dropout[i]`` decides the dropout probability on the output of the i-th GCN layer.
        ``len(dropout)`` equals the number of GCN layers. By default, no dropout is
        performed for all layers.
    allow_zero_in_degree: bool
        Whether to allow zero in degree nodes in graph for all layers. By default, will not
        allow zero in degree nodes.
    
    """

    def __init__(self, in_feats, hidden_feats, activation, dropout):
        super().__init__()
        self.gcn = GCN(
            in_feats=in_feats,
            hidden_feats=hidden_feats,
            activation=activation,
            dropout=dropout
        )
        gcn_out_feats = self.gcn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gcn_out_feats)

    def forward(self, graph, atom_feats):
        """
        Args:
            graph (DGLGraph): The input graph.
            atom_feats (torch.Tensor): The atom features of the graph.

        Returns:
            graph_feats (torch.Tensor): The graph-level features. [batch_size, 2 * hidden_feats[-1]]
        """
        node_feats = self.gcn(graph, atom_feats)
        graph_feats = self.readout(graph, node_feats)
        return graph_feats
    

class DNN(nn.Module):
    """
    Parameters
    ----------
    input_dim : int
        Number of input dimensions.
    hidden_dims : list
        Number of dimensions for each hidden layer.
    output_dim : int
        Number of output dimensions.
    dropouts : list
        Dropout probability for each layer.
    """

    def __init__(self, input_dim, hidden_dims, output_dim, dropout):
        super().__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim, drop in zip(hidden_dims, dropout):
            layers.append(nn.Linear(in_dim, hidden_dim, bias=False))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(drop))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)
        self.apply(self._init_weights)  # initialize weights and biases

    def forward(self, x):
        return self.model(x)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Kaiming initialization (recommended for ReLU activations)
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

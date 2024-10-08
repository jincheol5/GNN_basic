import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class EvolveGCN_H(nn.Module):
    def __init__(self, in_feats, out_feats, num_layers, device='cpu'):
        super(EvolveGCN_H, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.device = device
        
        # GRU for evolving GCN weights
        self.gru = nn.GRU(in_feats, out_feats, num_layers=num_layers, batch_first=True)
        
        # Initial GCN layer with evolving weights
        self.gcn = GCNConv(in_feats, out_feats)
        
        # Store GCN weights (initially random, will be learned over time)
        self.gcn_weights = torch.nn.Parameter(torch.randn(out_feats, in_feats).to(self.device))

    def forward(self, A_list, node_feats_list):
        """
        A_list: List of adjacency matrices (time series of adjacency matrices)
        node_feats_list: List of node features (time series of node features)
        """
        # Initialize GCN weights
        gcn_weights = self.gcn_weights
        out_feats_list = []
        
        # Loop through each time step
        for t in range(len(A_list)):
            A_t = A_list[t]  # Adjacency matrix at time step t
            node_feats_t = node_feats_list[t]  # Node features at time step t
            
            # Reshape GCN weights to match GRU input (batch, sequence, features)
            gcn_weights_input = gcn_weights.unsqueeze(0).unsqueeze(0)  # Add batch & sequence dimensions
            
            # Pass GCN weights through GRU to evolve weights over time
            evolved_weights, _ = self.gru(gcn_weights_input)
            evolved_weights = evolved_weights.squeeze(0).squeeze(0)  # Remove batch & sequence dimensions
            
            # Apply GCN with the evolved weights
            out_feats_t = self.gcn(node_feats_t, A_t)
            out_feats_list.append(out_feats_t)
            
            # Update GCN weights for the next time step
            gcn_weights = evolved_weights
        
        return out_feats_list
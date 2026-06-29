检查有没有问题：import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool

class SAGPoolLayer(nn.Module):
    def __init__(self, hidden_channels, keep_ratio):
        super().__init__()
        self.ratio = keep_ratio
        # Formula (11): Unique attention weight Θ_att
        self.att_gcn = GCNConv(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        num_total_nodes = x.size(0)

        # Step 1: Compute node importance score Z (Formula 11)
        # Z = σ( D^(-1/2) Â D^(-1/2) X Θ_att )
        z = self.att_gcn(x, edge_index)
        z = torch.sigmoid(z)

        # Step 2: Top-K hard node selection
        k = max(1, int(num_total_nodes * self.ratio))
        topk_idx = torch.topk(z.view(-1), k=k).indices

        # Step 3: Filter node features (Formula 6)
        x_filtered = x[topk_idx]
        z_filtered = z[topk_idx]

        # Step 4: Rebuild adjacency matrix (Formula 7)
        row, col = edge_index
        edge_mask = torch.isin(row, topk_idx) & torch.isin(col, topk_idx)
        row_new = row[edge_mask]
        col_new = col[edge_mask]

        # Remap node IDs after pooling
        idx_mapping = torch.full((num_total_nodes,), -1, dtype=torch.long, device=x.device)
        idx_mapping[topk_idx] = torch.arange(k, device=x.device)
        row_new = idx_mapping[row_new]
        col_new = idx_mapping[col_new]
        edge_new = torch.stack([row_new, col_new], dim=0)

        # Step 5: Feature gating with tanh (Formula 8)
        x_out = x_filtered * torch.tanh(z_filtered)
        batch_out = batch[topk_idx]

        # Step 6: Layer-wise readout: mean || max (Formula 9)
        feat_mean = global_mean_pool(x_out, batch_out)
        feat_max = global_max_pool(x_out, batch_out)
        layer_readout = torch.cat([feat_mean, feat_max], dim=-1)

        return x_out, edge_new, batch_out, layer_readout


class HierarchicalSAGPoolWDN(nn.Module):
    def __init__(self, node_input_dim, hidden_dim, num_class, pool_ratios=[0.3, 0.4, 0.6, 0.7]):
        super().__init__()
        self.pool_ratios = pool_ratios
        self.layer_num = len(pool_ratios)

        self.gcn_feature_blocks = nn.ModuleList()
        self.pool_blocks = nn.ModuleList()
        prev_dim = node_input_dim

        for _ in range(self.layer_num):
            # Formula (10): Standard GCN for feature extraction
            self.gcn_feature_blocks.append(GCNConv(prev_dim, hidden_dim))
            # Formula (11): SAGPool with GCN-att only
            self.pool_blocks.append(SAGPoolLayer(hidden_dim, keep_ratio=pool_ratios[_]))
            prev_dim = hidden_dim

        total_readout_dim = hidden_dim * 2 * self.layer_num
        self.class_head = nn.Sequential(
            nn.Linear(total_readout_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, num_class)
        )

    def forward(self, x, edge_index, batch):
        readout_buffer = []
        x_cur, edge_cur, batch_cur = x, edge_index, batch

        # Pipeline matches paper Figure 5:
        # Feature GCN(10) → GCN-Att SAGPool(11) → Readout, stacked 4 times
        for feat_gcn, sag_pool in zip(self.gcn_feature_blocks, self.pool_blocks):
            x_cur = feat_gcn(x_cur, edge_cur)
            x_cur = F.relu(x_cur)
            x_cur, edge_cur, batch_cur, layer_read = sag_pool(x_cur, edge_cur, batch_cur)
            readout_buffer.append(layer_read)

        global_graph_feat = torch.cat(readout_buffer, dim=-1)
        logits = self.class_head(global_graph_feat)
        return logits

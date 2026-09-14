"""
Causal Temporal Convolutional Network (TCN) with hierarchical attention mechanism.
--> TCN with learned attention. Attention can be causal or non-causal.

Based on
- Temporal Context Matters: Enhancing Single Image Prediction with Disease Progression Representations
  (Konwer et al., 2022) arXiv:2203.01933 (TCN with hierarchical attention)
- An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
  (Bai et al., 2018) arXiv:1803.01271 (TCN)
- Attention is all you need (Vaswani et al., 2017) (Attention)
- Longformer: The Long-Document Transformer (Beltagy et al., 2020) (Sliding window attention)

Related to:
-  TCNCA: Temporal Convolution Network with Chunked Attention for Scalable Sequence Processing
    (Terzic et al., 2023) arXiv:2312.05605 (TCN with chunked attention)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

class Chomp1d(nn.Module):
    # Removes padding from the end of convolution outputs to ensure that information doesn't leak from future timesteps to past ones
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention block for TCN layers; Non-causal attention"""
    
    def __init__(self, in_channels, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        
        # 1x1 convolutions for query, key, value transformations
        self.query_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=-1)
        self.scale = (self.head_dim) ** -0.5
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, channels, sequence_length)
        Returns:
            output: Attended features + residual connection
            attention_weights: Attention matrix for hierarchical aggregation
        """
        batch_size, channels, seq_len = x.size()
        
        # Transform to query, key, value
        f = self.query_conv(x)  # query
        g = self.key_conv(x)    # key  
        h = self.value_conv(x)  # value
        
        # Reshape for multi-head attention
        f = f.view(batch_size, self.num_heads, self.head_dim, seq_len)
        g = g.view(batch_size, self.num_heads, self.head_dim, seq_len)
        h = h.view(batch_size, self.num_heads, self.head_dim, seq_len)
        
        # Compute attention weights: A = softmax(f^T * g)
        # f^T: (batch, heads, seq_len, head_dim)
        # g: (batch, heads, head_dim, seq_len)
        attention = torch.matmul(f.transpose(-2, -1), g) * self.scale  # (batch, heads, seq_len, seq_len)
        attention = self.softmax(attention)
        
        # Apply attention to values: A^T * h
        attended = torch.matmul(attention, h.transpose(-2, -1))  # (batch, heads, seq_len, head_dim)
        attended = attended.transpose(-2, -1).contiguous()  # (batch, heads, head_dim, seq_len)
        
        # Concatenate heads
        attended = attended.view(batch_size, channels, seq_len)
        
        # Residual connection: o = x + A^T * h
        output = x + attended
        
        # Average attention across heads for hierarchical aggregation
        avg_attention = attention.mean(dim=1)  # (batch, seq_len, seq_len)
        
        return output, avg_attention

class CausalMultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention block for TCN layers; Causal attention with no attention
    to future positions.
    If window_size is None, can attend to all previous positions (including itself).
    If window_size is an integer, can attend to a local window (inclusive of current position) of 
    size window_size into the past.

    Based on 
        - full causal attention: Vaswani et al., 2017
            "We also modify the self-attention sub-layer in the decoder stack to prevent positions 
            from attending to subsequent positions"
        - sliding window attention: Beltagy et al., 2020
    """
    def __init__(self, in_channels, num_heads=8, window_size=None):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.window_size = window_size  # None = full causal, int = local window
        
        self.query_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        batch_size, channels, seq_len = x.size()
        
        q = self.query_conv(x).view(batch_size, self.num_heads, self.head_dim, seq_len)
        k = self.key_conv(x).view(batch_size, self.num_heads, self.head_dim, seq_len)
        v = self.value_conv(x).view(batch_size, self.num_heads, self.head_dim, seq_len)
        
        # Causal attention computation
        scores = torch.matmul(q.transpose(-2, -1), k) * self.scale
        
        # Create causal mask
        if self.window_size is None:
            # Full causal mask
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        else:
            # Sliding window causal mask
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            mask = mask * torch.triu(torch.ones(seq_len, seq_len, device=x.device), 
                                   diagonal=-self.window_size)
        
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention
        out = torch.matmul(attention, v.transpose(-2, -1))
        out = out.transpose(-2, -1).contiguous().view(batch_size, channels, seq_len)
        
        return x + out, attention.mean(dim=1)  # residual + averaged attention

class TemporalBlockWithAttention(nn.Module):
    """Temporal block with integrated self-attention (based on the locuslab TCN)"""
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, num_heads=8, causal=False):
        super(TemporalBlockWithAttention, self).__init__()
        
        # First convolution layer
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Self-attention block (inserted between conv layers as per paper)
        if causal:
            self.attention = CausalMultiHeadSelfAttention(n_outputs, num_heads)
        else:
            self.attention = MultiHeadSelfAttention(n_outputs, num_heads)
        
        # Second convolution layer
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection for the whole block
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # First conv + chomp + relu + dropout
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # Self-attention block
        out, attention_weights = self.attention(out)
        
        # Second conv + chomp + relu + dropout
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Residual connection for the entire block
        res = x if self.downsample is None else self.downsample(x)
        
        return self.relu(out + res), attention_weights

class HierarchicalTCN(nn.Module):
    """TCN with 'hierarchical' attention mechanism based on Konwer et al. 2022"""
    
    def __init__(self, num_inputs=512, num_channels=[512, 512, 512], kernel_size=3, 
                 dropout=0.2, num_heads=8, causal_attention=True):
        super(HierarchicalTCN, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        # Build TCN layers with attention blocks
        # Using dilations [1, 2, 4] as mentioned in Konwer et al. 2022
        for i in range(num_levels):
            dilation_size = 2 ** i  # This gives [1, 2, 4] for 3 layers
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Create temporal block with attention
            layers += [TemporalBlockWithAttention(
                in_channels, out_channels, kernel_size, stride=1, 
                dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size, 
                dropout=dropout, num_heads=num_heads,
                causal=causal_attention  # Pass causal parameter
            )]
        
        self.network = nn.ModuleList(layers)
        self.num_levels = num_levels
        
    def forward(self, x):
        """
        Args:
            x: Input sequence of shape (batch_size, num_inputs, sequence_length)
            
        Returns:
            optimal_representation: Single vector per sequence (batch_size, num_channels[-1])
            all_attention_weights: List of attention matrices from each layer
        """
        all_attention_weights = []
        
        # Forward through all TCN layers
        for layer in self.network:
            x, attention_weights = layer(x)
            all_attention_weights.append(attention_weights)
        
        # Hierarchical attention aggregation to get optimal representation
        optimal_representation = self._aggregate_temporal_representation(x, all_attention_weights)
        
        return optimal_representation, all_attention_weights
    
    def _aggregate_temporal_representation(self, features, attention_weights_list):
        """
        Aggregate attention weights across layers to find optimal representation
        Following the Konwer et al. 2022's description of hierarchical attention
        
        Args:
            features: Final layer features (batch_size, channels, seq_len)
            attention_weights_list: List of attention matrices from each layer
        """
        batch_size, channels, seq_len = features.size()
        
        # Sum attention weights across all layers (as mentioned in paper)
        # Each attention matrix is (batch_size, seq_len, seq_len)
        combined_attention = torch.zeros_like(attention_weights_list[0])
        
        for attention in attention_weights_list:
            combined_attention += attention
        
        # Row-wise summation to get importance score for each timepoint
        # Paper: "We sum up each row of A to get a T dimensional weight vector α, 
        # measuring the contribution of each time slice to all other time slices"
        timepoint_importance = combined_attention.sum(dim=2)  # (batch_size, seq_len)
        
        # Apply softmax to get normalized global attention weights
        global_attention = F.softmax(timepoint_importance, dim=1)  # (batch_size, seq_len)
        
        # Weighted sum of features across time dimension
        # Paper: "use the output to calculate a weighted sum of the representations at all time slices"
        global_attention = global_attention.unsqueeze(1)  # (batch_size, 1, seq_len)
        
        # Compute weighted representation - this is the "optimal" representation
        optimal_representation = torch.sum(features * global_attention, dim=2)  # (batch_size, channels)
        
        return optimal_representation

# Wrapper class for easy integration with the recalibration network
class TemporalFeatureExtractor(nn.Module):
    """Complete temporal feature extraction pipeline"""
    
    def __init__(self, input_dim=512, hidden_channels=[512, 512, 512], kernel_size=3, 
                 dropout=0.2, num_heads=8, num_classes=None):
        super(TemporalFeatureExtractor, self).__init__()
        
        self.tcn = HierarchicalTCN(
            num_inputs=input_dim,
            num_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            num_heads=num_heads
        )
        
        # Optional classification head for pretraining
        if num_classes is not None:
            self.classifier = nn.Linear(hidden_channels[-1], num_classes)
        else:
            self.classifier = None
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: Input tensor (batch_size, input_dim, sequence_length)
            return_attention: Whether to return attention weights
            
        Returns:
            If classifier exists: logits for pretraining
            If no classifier: optimal temporal representation for recalibration
        """
        optimal_repr, attention_weights = self.tcn(x)
        
        if self.classifier is not None:
            # For pretraining phase
            logits = self.classifier(optimal_repr)
            if return_attention:
                return logits, optimal_repr, attention_weights
            return logits
        else:
            # For recalibration phase - return just the representation
            if return_attention:
                return optimal_repr, attention_weights
            return optimal_repr

# Example usage and testing
# if __name__ == "__main__":
#     # Test the hierarchical attention TCN
#     batch_size = 8
#     seq_len = 10  # T timepoints
#     input_channels = 512  # ResNet-18 feature dimension
    
#     # Create sample input (batch_size, channels, sequence_length)
#     sample_input = torch.randn(batch_size, input_channels, seq_len)
    
#     print(f"Input shape: {sample_input.shape}")
    
#     # Test 1: TCN without classifier (for recalibration)
#     model_recalibration = TemporalFeatureExtractor(
#         input_dim=512,
#         hidden_channels=[512, 512, 512],
#         kernel_size=3,
#         dropout=0.2,
#         num_heads=8,
#         num_classes=None  # No classifier for recalibration
#     )
    
#     optimal_repr = model_recalibration(sample_input)
#     print(f"Optimal representation shape: {optimal_repr.shape}")
#     assert optimal_repr.shape == (batch_size, 512), "Output should be (batch_size, 512)"
    
#     # Test 2: TCN with classifier (for pretraining)
#     model_pretraining = TemporalFeatureExtractor(
#         input_dim=512,
#         hidden_channels=[512, 512, 512],
#         kernel_size=3,
#         dropout=0.2,
#         num_heads=8,
#         num_classes=5  # For classification pretraining
#     )
    
#     logits = model_pretraining(sample_input)
#     print(f"Pretraining logits shape: {logits.shape}")
#     assert logits.shape == (batch_size, 5), "Logits should be (batch_size, num_classes)"
    
#     # Test 3: Get attention weights
#     optimal_repr, attention_weights = model_recalibration(sample_input, return_attention=True)
#     print(f"Number of attention layers: {len(attention_weights)}")
#     print(f"Attention weight shape per layer: {attention_weights[0].shape}")
    
#     print("✓ All tests passed!")
#     print("✓ Implementation follows standard TCN structure with hierarchical attention")
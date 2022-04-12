import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

def sequence_mask(lengths, maxlen, dtype=torch.bool):
    """Copied from https://discuss.pytorch.org/t/pytorch-equivalent-for-tf-sequence-mask/39036/3"""
    if maxlen is None:
        maxlen = lengths.max()
    mask = ~(torch.ones((len(lengths), maxlen)).cumsum(dim=1).t() > lengths).t()
    mask.type(dtype)
    return mask

class Wider_Actor(nn.Module):
    r"""Net2Wider Network.
    
    Shape:
    - Input: `(batch_size * num_steps, in_dim)` where `in_dim` means `rnn_units` the dimension of output from bi-LSTM network
    - Output: a tuple containing decision `(batch_size, num_steps)`, and probs `(batch_size, num_steps, out_dim)` where `batch_size`

    Return:
        a tuple of two tensors (decision, probs)

    Args:
    - `in_dim` (int): the input dimension, namely `rnn_units`
    - `out_dim` (int): the output dimension of the intermediate fully connected layer, by default is 1
    - `num_steps` (int): the length of the input network string sequence
    - `net_config` (dict): remains for future tuning, by default is `None`

    """
    def __init__(self, in_dim, out_dim=1, net_config=None) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.net_config = net_config

        self.fc = nn.Linear(in_dim, out_dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, num_steps):
        r"""Build forward for Net2Wider network.
        
        Args:
        - `x`: input tensor with shape (batch_size * num_steps, in_dim)

        """
        out1 = self.fc(x)
        out2 = self.sigmoid(out1)
        # out2: a tensor with shape (batch_size * num_steps, 1) by default out_dim
        if self.out_dim == 1:
            probs = torch.concat((1 - out2, out2), dim=1)
            probs_dim = 2
        else:
            raise NotImplementedError
        decision = torch.multinomial(probs, 1)
        # decision: shape becomes (batch_size * num_steps, 1)
        decision = torch.reshape(decision, (-1, num_steps))
        # decision: shape becomes (batch_size, num_steps)
        probs = torch.reshape(probs, (-1, num_steps, probs_dim))
        # probs: shape becomes (batch_size, num_steps, 2)

        return (decision, probs)

class Deeper_Actor(nn.Module):
    """Net2Deeper Network.
    
    Shape:
    - Input: Tuple, each entry has shape `(batch_size, in_dim)` where `in_dim` concats forward and backward rnn_units from encoder output state
    - Output: a tuple containing decision `(batch_size, num_steps)`, and probs `(batch_size, num_steps, out_dim)` where `batch_size`

    Return:
        a tuple of two tensors (decision, probs)

    Args:
    - `in_dim` (int): the input dimension, namely `rnn_units`
    
    """
    def __init__(self, embedding_dim: int, hidden_size: int, out_dims: List[int]=[16, 4, 5], num_lstms: int=3) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        # self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_dims = out_dims
        self.num_lstms = num_lstms # layer, filter size, stride

        self.embeddings = nn.ModuleList()
        self.lstms = nn.ModuleList()
        self.linears = nn.ModuleList()
        
        # input_size, hidden_size = self.input_size, self.hidden_size
        for i in range(num_lstms):
            # self.out_dims[i]: the vocabulary size of the embedding
            if i == 0:
                self.embeddings.append(nn.Embedding(self.out_dims[i], self.embedding_dim))
            else:
                self.embeddings.append(nn.Embedding(self.out_dims[i-1], self.embedding_dim))
            self.lstms.append(nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_size, num_layers=1, bias=True))
            self.linears.append(nn.Linear(self.hidden_size, self.out_dims[i], bias=True))

            # TODO: check input_size
            # input_size = hidden_size

            

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor], is_training: bool=True, decision_trajectory=None):
        """Build forward for Net2Deeper network.
        
        Args:
        - `x`: tuple of input tensor with shape (batch_size, in_dim), which contains the encoder state (h, c)
        - `is_training`: optional, by default is True
        """

        batch_size = x[0].shape[0]
        # the input of LSTM before embedding
        cell_input = torch.zeros(batch_size, dtype=torch.int32)
        # reshape the encoder_state x to meet the requirement of LSTM input
        reshape_x = (torch.reshape(x[0], (1, *map(int, x[0].shape))),
            torch.reshape(x[1], (1, *map(int, x[1].shape))))
        decision, probs = list(), list()
        for i in range(self.num_lstms):
            cell_input_embed = self.embeddings[i](cell_input)
            cell_input_embed = cell_input_embed.reshape(1, *map(int, cell_input_embed.shape))
            # sequence length=1, need reshape
            # cell input: (1, batch_size, input_size)
            # h_0, c_0, h_1, c_1: (1, batch_size, hidden_size)
            cell_output, reshape_x = self.lstms[i](cell_input_embed, reshape_x)

            cell_output = torch.reshape(cell_output, (cell_output.shape[1], -1))
            logits_i = self.linears[i](cell_output)
            probs_i = F.softmax(logits_i, dim=1) # (batch, out_dim[i])
            # choose the sampled result
            decision_i = torch.multinomial(probs_i, 1) # (batch, 1)
            decision_i = torch.reshape(decision_i, (-1, )) # (1, batch)
            # update cell_input in the for loop, make decision 
            if is_training:
                cell_input = decision_trajectory[:, i]
            else:
                cell_input = decision_i
            decision.append(decision_i)
            probs.append(probs_i) # (len(out_dim), batch, out_dim[i])
        decision = torch.stack(decision, dim=1) # (batch, sum(out_dim))
        return (decision, probs)

class Encoder_Network(nn.Module):
    def __init__(self, vocab_size: int = 12, embedding_dim: int = 16, hidden_dim: int = 50) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True)

    def forward(self, x):
        """
        shape of x: batch, seq_len
        """
        x = self.embedding(x)
        x = x.transpose(0,1)
        lstm_out, (h, c) = self.lstm(x)
        return lstm_out, (h, c)

class Policy_Network(nn.Module):
    def __init__(self, 
                 encoder_vocab_size: int, # 12
                 encoder_embedding_dim: int, # 16
                 encoder_hidden_dim: int, # 50
                 deeper_embedding_dim: int, # 16
                 deeper_out_dims: List,
                 wider_out_dim: int=1,
                 is_training: bool=False
                ) -> None:
        super().__init__()
        self.encoder = Encoder_Network(encoder_vocab_size, encoder_embedding_dim, encoder_hidden_dim)
        self.deeper = Deeper_Actor(deeper_embedding_dim, encoder_hidden_dim*2, out_dims=deeper_out_dims)
        self.wider = Wider_Actor(encoder_hidden_dim*2, out_dim=wider_out_dim)
        self.deeper_hidden_dim = encoder_hidden_dim*2

    def forward(self, x, num_steps, is_training: bool, decision_trajectory=None):
        x, (h, c) = self.encoder(x)
        h = torch.reshape(h, (h.shape[1], -1))
        c = torch.reshape(c, (c.shape[1], -1))
        deeper_decision, deeper_probs = self.deeper((h, c), is_training, decision_trajectory=decision_trajectory) # not sure
        x = torch.reshape(x, (-1, x.shape[2]))
        wider_decision, wider_probs = self.wider(x, num_steps)
        return deeper_decision, deeper_probs, wider_decision, wider_probs
    
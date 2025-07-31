"""
Modified or original code

"""

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math

    
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self,device):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class TransRNN(nn.Module):
    def __init__(self,input_size, hidden_size, output_size, device, context_len):
        super(TransRNN, self).__init__()
        self.encoder = EncoderRNN(input_size,hidden_size)
        self.decoder = DecoderRNN(hidden_size,output_size)
        self.context_len = context_len
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device

    def forward(self, src, tgt): 
        encoder_hidden = self.encoder.initHidden(self.device)
        encoder_outputs = torch.zeros(src.shape[0], 1, self.hidden_size)
        decoder_output = torch.zeros(src.shape[0], self.context_len, self.output_size)
        for ei in range(self.context_len):
            encoder_outputs[:,0,:], encoder_hidden = self.encoder(src[:,ei,:], encoder_hidden)
        
        decoder_hidden = encoder_outputs

        for di in range(self.context_len):
            decoder_output[:,di,:], decoder_hidden = self.decoder(tgt[:,di,:], decoder_hidden)
        
        return decoder_output
    
    def translate(self,src,tgt,tr_set_stats, target_state):
        self.eval()

        with torch.no_grad():
            encoder_hidden = self.encoder.initHidden(self.device)
            encoder_outputs = torch.zeros(1, 1, self.hidden_size)
            decoder_output = torch.zeros(1, self.context_len, self.output_size)

            for ei in range(self.context_len):
                encoder_outputs[:,0,:], encoder_hidden = self.encoder(src[:,ei,:], encoder_hidden)
        
            decoder_hidden = encoder_outputs
            decoder_input = tgt[:,0,:]
            path_length = 0

            for di in range(self.context_len):
                decoder_output[:,di,:], decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                decoder_input = decoder_output[:,di,:]
                mean, std = tr_set_stats
                txy = decoder_input.reshape(decoder_input.shape[-1])
                txy = txy * std + mean
                if torch.norm((txy[1:]-target_state[0,1:])) <= 2:
                    break

                path_length += 1

            return decoder_output, path_length
        
        
            
class TrajectoryModel(nn.Module):

    def __init__(self, state_dim, act_dim, max_length=None):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

    def forward(self, states, actions, rewards, masks=None, attention_mask=None):
        # "masked" tokens or unspecified inputs can be passed in as None
        return None, None, None

    def get_action(self, states, actions, rewards, **kwargs):
        # these will come as tensors on the correct device
        return torch.zeros_like(actions[-1])
    
    
    
class MLPBCModel(nn.Module):

    """
    Simple MLP that predicts next action a from past states s.
    """

    def __init__(self, state_dim, act_dim, hidden_size, n_layer, dropout=0.1, max_length=1, **kwargs):
        super().__init__()

        self.hidden_size = hidden_size
        self.max_length = max_length
        self.state_dim = state_dim
        self.act_dim = act_dim
        
        layers = [nn.Linear(max_length*self.state_dim, hidden_size)]
        for _ in range(n_layer-1):
            layers.extend([
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size)
            ])
        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, max_length*self.act_dim),
            nn.Sigmoid(),
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, states):
        # states shape: B, T, s_dim
        states = states.reshape(states.shape[0], -1)  # concat states
        
        actions = self.model(states).reshape(states.shape[0], self.max_length, self.act_dim)
        # actions shape: B, T, a_dim
        return actions

    # def get_action(self, states, actions, rewards, **kwargs):
    #     states = states.reshape(1, -1, self.state_dim)
    #     if states.shape[1] < self.max_length:
    #         states = torch.cat(
    #             [torch.zeros((1, self.max_length-states.shape[1], self.state_dim),
    #                          dtype=torch.float32, device=states.device), states], dim=1)
    #     states = states.to(dtype=torch.float32)
    #     _, actions, _ = self.forward(states, None, None, **kwargs)
    #     return actions[0,-1]





# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        # print(f"in posEnc.init emb_size = {emb_size}")

        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        # pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor, timesteps: Tensor):
        # print(f"in posEnc.forward token_embedding.shape = {token_embedding.shape},\n self.pos_embedding.shape = {self.pos_embedding.shape}, {token_embedding.size(0)}")
        emb = self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(1), :])
        return emb
                                # [64, 70, 8]            (70,8)

class SimplePositionalEncoding(nn.Module):
    def __init__(self, emb_size, max_len):
        super(SimplePositionalEncoding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, emb_size)

    def forward(self, emb_vec, timesteps):
        return self.pos_embedding(timesteps) + emb_vec

# # helper Module to convert tensor of input indices into corresponding tensor of token embeddings
# class TokenEmbedding(nn.Module):
#     def __init__(self, vocab_size: int, emb_size):
#         super(TokenEmbedding, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, emb_size)
#         self.emb_size = emb_size

#     def forward(self, tokens: Tensor):
#         return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class LinTokenEmbedding(nn.Module):
    def __init__(self, ip_vec_dim: int, emb_size):
        super(LinTokenEmbedding, self).__init__()
        self.embedding = nn.Linear(ip_vec_dim, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.to(torch.float32)) * math.sqrt(self.emb_size)  

# Seq2Seq Network
class mySeq2SeqTransformer_bc(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                #  num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vec_dim: int,
                 tgt_vec_dim: int,
                 dim_feedforward: int = None,
                 dropout: float = 0.1,
                 max_len: int = 120,
                 batch_first = True,
                 positional_encoding: str = "simple",
                 ):
        super(mySeq2SeqTransformer_bc, self).__init__()

        if dim_feedforward == None:
            dim_feedforward = nhead * emb_size
 
        self.max_len = max_len
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_encoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=batch_first)
        self.generator = nn.Sequential(nn.Linear(emb_size, tgt_vec_dim), nn.Sigmoid())
        self.src_tok_emb = LinTokenEmbedding(src_vec_dim, emb_size)
        self.tgt_tok_emb = LinTokenEmbedding(tgt_vec_dim, emb_size)
        if positional_encoding == 'sin':
            self.positional_encoding = PositionalEncoding(
                emb_size, dropout=dropout, maxlen=max_len)
        elif positional_encoding == "simple":
            self.positional_encoding = SimplePositionalEncoding(emb_size, max_len)
        else:
            raise ValueError("No such positional_encoding type")

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor,
                timesteps: Tensor):
        # print(f"*** in myseq.forward src.shape = {src.shape},{src.dtype}, {trg.shape}, {trg.dtype} ")
        #  # positional embedding usage as in pytorch translation tutorial
        # src_emb = self.positional_encoding(self.src_tok_emb(src))
        # tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        
        #  simple positional embedding as in Decistion Transformer
        src_emb = self.positional_encoding(self.src_tok_emb(src), timesteps)
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg), timesteps)
        encoder_output = self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src), timesteps), src_mask)
        # print(f"src and tgt shapes ; {src_emb.shape}, {tgt_emb.shape}")
        # outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
        #                         src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(encoder_output)

    # def encode(self, src: Tensor, src_mask: Tensor, timesteps: Tensor):
    #     return self.transformer.encoder(self.positional_encoding(
    #                         self.src_tok_emb(src), timesteps), src_mask)




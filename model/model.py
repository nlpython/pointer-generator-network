import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model.model_utils import sort_batch_by_length


class SummaryModel(nn.Module):

    def __init__(self, args):
        super(SummaryModel, self).__init__()

        self.args = args

        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

        self.reduce_state = ReduceState(args)

        # share weight
        # self.decoder.embedding.weight = self.encoder.embedding.weight


class Encoder(nn.Module):

    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args

        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.lstm = nn.LSTM(args.embedding_dim, args.hidden_dim,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.W_h = nn.Linear(args.hidden_dim*2, args.hidden_dim*2, bias=False)

        self.init_weight()


    def init_weight(self):
        self.embedding.weight.data.normal_(std=self.args.embedding_init_std)

        for name, parm in self.lstm.named_parameters():
            if name.startswith('weight_'):
                nn.init.xavier_uniform_(parm)
                # parm.data.weight.uniform_(-self.args.lstm_init_range, self.args.lstm_init_range)
            elif name.startswith('bias_'):
                n = parm.data.size(0)
                start, end = n // 4, n // 2
                parm.data.data.fill_(0)
                parm.data.data[start:end].fill_(1)


    def forward(self, input, seq_lens):
        embedded = self.embedding(input)

        (sorted_input, sorted_lengths, input_unsort_indices, _) = sort_batch_by_length(embedded, seq_lens)
        packed = pack_padded_sequence(sorted_input, sorted_lengths.cpu().numpy(), batch_first=True)
        output, (sorted_h_n, sorted_c_n) = self.lstm(packed)
        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        encoder_outputs = encoder_outputs[input_unsort_indices]
        encoder_outputs = encoder_outputs.contiguous()

        encoder_features = encoder_outputs.view(-1, self.args.hidden_dim*2)  # B * t_k x hidden_dim*2
        encoder_features = self.W_h(encoder_features)   # [B*length, hidden_dim*2]

        h_n = sorted_h_n[:, input_unsort_indices]   # [layer_num*direction_num(2), B, hidden_dim]
        c_n = sorted_c_n[:, input_unsort_indices]   # [layer_num*direction_num(2), B, hidden_dim]
        hidden = (h_n, c_n)

        return encoder_outputs, encoder_features, hidden


class Attention(nn.Module):

    def __init__(self, args):
        super(Attention, self).__init__()
        self.args = args

        if args.use_coverage:
            self.W_c = nn.Linear(1, args.hidden_dim*2, bias=False)

        self.decode_proj = nn.Linear(args.hidden_dim*2, args.hidden_dim*2)
        self.v = nn.Linear(args.hidden_dim*2, 1, bias=False)

    def forward(self, s_t, encoder_outputs, encoder_features, enc_padding_mask, coverage):
        """
        Args:
            s_t: [B, 2*H]
            encoder_outputs: [B, L, 2*H]
            encoder_features: [B*L, 2*H]
            enc_padding_mask: [B, L]
            coverage: None or [B, L]
        Returns:
            c_t: [B, 2*H]
            att_list: [B, L]
            coverage: None or [B, L]
        """
        b, t_k, n = list(encoder_outputs.size())    # B, L, 2*H

        dec_features = self.decode_proj(s_t)    # [B, 2*H]
        dec_features_expanded = dec_features.unsqueeze(1).expand(b, t_k, n).contiguous() # [B, L, 2*H]
        dec_features_expanded = dec_features_expanded.view(-1, n)  # [B*L, 2*H]

        att_features = encoder_features + dec_features_expanded  # [B*L, 2*H]

        if self.args.use_coverage:
            coverage_input = coverage.view(-1, 1)
            coverage_feature = self.W_c(coverage_input)
            att_features = att_features + coverage_feature

        e = torch.tanh(att_features)  # [B*L, 2*H]
        scores = self.v(e)  # [B*L, 1]
        scores = scores.view(-1, t_k)  # [B, L]

        att_dist = torch.softmax(scores, dim=1) * enc_padding_mask  # [B, L]
        normalization_factor = att_dist.sum(dim=1, keepdim=True)    # [B, 1]
        att_dist = att_dist / normalization_factor   # [B, L]

        att_dist = att_dist.unsqueeze(1)  # [B, 1, L]
        c_t = torch.bmm(att_dist, encoder_outputs)  # [B, 1, 2*H]
        c_t = c_t.view(-1, self.args.hidden_dim*2)  # [B, 2*H]

        att_dist = att_dist.view(-1, t_k)  # [B, L]

        if self.args.use_coverage:
            coverage = coverage.view(-1, 1, t_k)
            coverage = coverage + att_dist

        return c_t, att_dist, coverage



class Decoder(nn.Module):

    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args

        self.attention = Attention(args)

        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.x_content = nn.Linear(args.hidden_dim*2 + args.embedding_dim, args.embedding_dim)

        self.lstm = nn.LSTM(args.embedding_dim, args.hidden_dim,
                            num_layers=1, batch_first=True, bidirectional=False)

        if args.use_pointer_gen:
            self.p_gen_linear = nn.Linear(args.hidden_dim * 4 + args.embedding_dim, 1)

        # p_vocab
        self.out1 = nn.Linear(args.hidden_dim * 3, args.hidden_dim)
        self.out2 = nn.Linear(args.hidden_dim, args.vocab_size)

        self.init_weight()

    def init_weight(self):
        self.embedding.weight.data.normal_(std=self.args.embedding_init_std)
        self.out2.weight.data.normal_(std=self.args.embedding_init_std)

        for name, parm in self.lstm.named_parameters():
            if name.startswith('weight_'):
                nn.init.xavier_uniform_(parm)
                # parm.data.weight.uniform_(-self.args.lstm_init_range, self.args.lstm_init_range)
            elif name.startswith('bias_'):
                n = parm.data.size(0)
                start, end = n // 4, n // 2
                parm.data.data.fill_(0)
                parm.data.data[start:end].fill_(1)

    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_features, enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extended_vocab, coverage, step):
        """
        Args:
            y_t_1: [B,]
            s_t_1: (h_t, c_t) [1, B, H]
            encoder_outputs: [B, L, H*2]
            encoder_features: [B*L, H*2]
            enc_padding_mask: [B, L]
            c_t_1: [B, H*2]
            extra_zeros: [B, max_len of OOVs in current batch]
            enc_batch_extended_vocab: [B, L]
            coverage: None
            step: 0 ~ decoder_step - 1
        Returns:
            final_dist: [B, vocab_size+max_len_oovs]
            s_t: (h_t, c_t) [1, B, H]
            c_t: [B, H*2]
            att_dist: [B, L]
            p_gen: [B, 1]
            coverage: None or [B, L]
        """
        if not self.training and step == 0:  # inference and first step
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, self.args.hidden_dim),
                                 c_decoder.view(-1, self.args.hidden_dim)), 1)  # B x 2*hidden_dim

            c_t, _, coverage_next = self.attention(s_t_hat, encoder_outputs, encoder_features,
                                                   enc_padding_mask, coverage)

            coverage = coverage_next

        y_t_1_embed = self.embedding(y_t_1) # [B, embedding_dim]
        x = self.x_content(torch.cat((c_t_1, y_t_1_embed), 1))  # [B, embedding_dim]
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)    # [B, 1, H], ([1, B, H]*2)

        h_decoder, c_decoder = s_t  # [1, B, H]
        s_t_hat = torch.cat((h_decoder.view(-1, self.args.hidden_dim),
                             c_decoder.view(-1, self.args.hidden_dim)), dim=1)  # [B, 2*H]
        # [B, H*2], [B, L], None or [B, L]
        c_t, attn_dist, coverage_next = self.attention(s_t_hat, encoder_outputs, encoder_features,
                                                       enc_padding_mask, coverage)

        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None
        if self.args.use_pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)   # [B, 2*2*H+embedding_dim]
            p_gen = self.p_gen_linear(p_gen_input)      # [B ,1]
            p_gen = torch.sigmoid(p_gen)                # [B ,1]

        output = torch.cat((lstm_out.view(-1, self.args.hidden_dim), c_t), 1)   # [B, H*3]
        output = self.out1(output)  # [B, H]

        output = self.out2(output)  # [B, vocab_size]
        vocab_dist = F.softmax(output, dim=1)   # [B, vocab_size]

        if self.args.use_pointer_gen:
            vocab_dist_ = p_gen * vocab_dist    # [B, vocab_size]
            attn_dist_ = (1 - p_gen) * attn_dist    # [B, L]

            if extra_zeros is not None:
                # print(vocab_dist_.size(), extra_zeros.size()) # todo wait delete
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], dim=1)  # [B, vocab_size+max_len_oovs]

            final_dist = vocab_dist_.scatter_add(1, enc_batch_extended_vocab, attn_dist_)   # [B, vocab_size+max_len_oovs]
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage


class ReduceState(nn.Module):
    def __init__(self, args):
        super(ReduceState, self).__init__()
        self.args = args

        self.reduce_h = nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        self.reduce_c = nn.Linear(args.hidden_dim * 2, args.hidden_dim)

        self.init_weight()

    def init_weight(self):
        self.reduce_h.weight.data.normal_(std=self.args.embedding_init_std)
        self.reduce_c.weight.data.normal_(std=self.args.embedding_init_std)

    def forward(self, hidden):
        h, c = hidden # [2, B, H]
        h_in = h.transpose(0, 1).contiguous().view(-1, self.args.hidden_dim * 2)  # [B, H*2]
        hidden_reduced_h = F.relu(self.reduce_h(h_in))  # [B, H]
        c_in = c.transpose(0, 1).contiguous().view(-1, self.args.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0)) # h, c dim = 1 x b x hidden_dim

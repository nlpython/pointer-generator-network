import os
import time

import torch
from loguru import logger
from torch.autograd import Variable
from processing.data_utils import Vocab, SummaryDataset
from processing.hyper_parm import print_arguments, get_parser
from processing.processor import Processor
from torch.utils.data import DataLoader
from processing.tools import outputids2words, decode_output_ids
from model.model import SummaryModel
from utils import rouge_log, rouge_eval, write_for_rouge

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args = get_parser()

def decode(model):


    vocab = Vocab(args.vocab_path, args.vocab_size, logger)
    processer = Processor(args, vocab, logger)
    dataset = SummaryDataset(processer.load_and_cache_examples('dev'))
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.collate_fn
    )

    model = model.to(device).eval()

    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch[:-3] if t is not None) + batch[-3:]
            enc_batch, enc_batch_len, enc_padding_mask, dec_batch, dec_batch_len, dec_padding_mask, \
            target_batch, enc_batch_extend_vocab, extra_zeros, content_oovs, title, content = batch

            c_t_1 = torch.zeros((enc_batch.shape[0], 2 * args.hidden_dim), device=device)

            coverage = None
            if args.use_coverage:
                coverage = torch.zeros((enc_batch.shape[0]), device=device)


            # [B*2, L, H], [B*L, H*2], ([2, B, H]*2)
            encoders_outputs, encoder_features, encoder_hidden = \
                model.encoder(enc_batch, enc_batch_len)

            s_t_1 = model.reduce_state(encoder_hidden)  # (h_t, c_t) [1, B, H]

            word_list = []

            # decode
            for di in range(args.max_dec_len):
                y_t_1 = dec_batch[:, di]  # Teacher forcing [B,]
                final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = \
                    model.decoder(y_t_1, s_t_1, encoders_outputs, encoder_features,
                                  enc_padding_mask, c_t_1, extra_zeros, enc_batch_extend_vocab,
                                  coverage, di)

                # target = target_batch[:, di]  # [B,]
                # gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze(1)  # [16,]
                # step_loss = -torch.log(gold_probs + 1e-10)

                word_id = torch.argmax(final_dist, 1).item()
                word = decode_output_ids(word_id, vocab, content_oovs[0])

                if word == '<EOS>':
                    break

                word_list.append(word)

            print('\nTarget: {}'.format(title[0]))
            print('Decode:', ''.join(word_list))





if __name__ == '__main__':
    # load model
    model = SummaryModel(args)
    model.load_state_dict(torch.load(f'./checkpoints/SummmaryModel_34.pth'))

    decode(model)



# Except for the pytorch part content of this file is copied from https://github.com/abisee/pointer-generator/blob/master/


import os
import time

import torch
from loguru import logger
from torch.autograd import Variable
from processing.data_utils import Vocab, SummaryDataset
from processing.hyper_parm import print_arguments, get_parser
from processing.processor import Processor
from torch.utils.data import DataLoader
from processing.tools import outputids2words
from model.model import SummaryModel
from utils import rouge_log, rouge_eval, write_for_rouge


args = get_parser()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class Beam(object):
    def __init__(self, tokens, log_probs, state, context, coverage):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context
        self.coverage = coverage

    def extend(self, token, log_prob, state, context, coverage):
        return Beam(tokens=self.tokens + [token],
                    log_probs=self.log_probs + [log_prob],
                    state=state,
                    context=context,
                    coverage=coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def __init__(self, model):
        self._decode_dir = os.path.join('./decode_results/')
        self._rouge_ref_dir = os.path.join(self._decode_dir, 'rouge_ref')
        self._rouge_dec_dir = os.path.join(self._decode_dir, 'rouge_dec_dir')
        for p in [self._decode_dir, self._rouge_ref_dir, self._rouge_dec_dir]:
            if not os.path.exists(p):
                os.mkdir(p)

        self.vocab = Vocab(args.vocab_path, args.vocab_size, logger=logger)
        self.processer = Processor(args, self.vocab, logger)
        dataset = SummaryDataset(self.processer.load_and_cache_examples('dev'))
        self.batcher = DataLoader(
            dataset=dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=dataset.collate_fn
        )
        time.sleep(3)

        self.model = model.to(device)
        self.model.eval()

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def decode(self):
        start = time.time()
        counter = 0
        for batch in self.batcher:
            # Run beam search to get best Hypothesis
            best_summary = self.beam_search(batch)

            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            decoded_words = outputids2words(output_ids, self.vocab,
                                             (batch[9] if args.use_pointer_gen else None))     ####

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(self.vocab.word2id('<EOS>'))
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            original_abstract_sents = batch[-2]
            print('original_abstract_sents: ', original_abstract_sents)
            print('Decode result:' , ''.join(decoded_words))

            write_for_rouge(original_abstract_sents, decoded_words, counter,
                            self._rouge_ref_dir, self._rouge_dec_dir)
            counter += 1
            if counter % 1000 == 0:
                print('%d example in %d sec' % (counter, time.time() - start))
                start = time.time()


        print("Decoder has finished reading dataset for single_pass.")
        print("Now starting ROUGE eval...")
        results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        rouge_log(results_dict, self._decode_dir)

    def beam_search(self, batch):
        # batch should have only one example
        global coverage_t_0
        batch = tuple(t.to(device) for t in batch[:-3] if t is not None) + batch[-3:]
        enc_batch, enc_batch_len, enc_padding_mask, dec_batch, dec_batch_len, dec_padding_mask, \
        target_batch, enc_batch_extend_vocab, extra_zeros, content_oovs, title, content = batch

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_batch_len)
        s_t_0 = self.model.reduce_state(encoder_hidden)

        c_t_0 = torch.zeros((enc_batch.shape[0], 2 * args.hidden_dim), device=device)
        if args.use_coverage:
            coverage_t_0 = torch.zeros((enc_batch.shape[0]), device=device)

        dec_h, dec_c = s_t_0  # 1 x 2*hidden_size
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        # decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[self.vocab.word2id('<SOS>')],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context=c_t_0[0],
                      coverage=(coverage_t_0[0] if args.use_coverage else None))
                 for _ in range(args.beam_size)]
        results = []
        steps = 0
        while steps < args.max_dec_len and len(results) < args.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id('<UNK>') \
                             for t in latest_tokens]
            y_t_1 = torch.LongTensor(latest_tokens).to(device)
            all_state_h = []
            all_state_c = []

            all_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)

                all_context.append(h.context)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)

            coverage_t_1 = None
            if args.use_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t_1, s_t_1,
                                                                                    encoder_outputs, encoder_feature,
                                                                                    enc_padding_mask, c_t_1,
                                                                                    extra_zeros, enc_batch_extend_vocab,
                                                                                    coverage_t_1, steps)
            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, args.beam_size * 2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if args.use_coverage else None)

                for j in range(args.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                        log_prob=topk_log_probs[i, j].item(),
                                        state=state_i,
                                        context=context_i,
                                        coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            min_dec_len = 3
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id('<EOS>'):
                    if steps >= min_dec_len:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == args.beam_size or len(results) == args.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0]


if __name__ == '__main__':

    # load model
    model = SummaryModel(args)
    model.load_state_dict(torch.load(f'./checkpoints/SummmaryModel_34.pth'))

    beam_Search_processor = BeamSearch(model)


    beam_Search_processor.decode()

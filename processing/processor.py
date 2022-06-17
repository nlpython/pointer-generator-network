import pickle
import json
import os
import pkuseg
from tqdm import tqdm
from processing.data_utils import Feature
from processing.hyper_parm import get_parser
from processing.tools import get_dec_input_and_target, title2ids, content2ids


class Processor(object):

    def __init__(self, args, vocab, logger=None):
        self.args = args
        self.vocab = vocab
        self.logger = logger


    def load_and_cache_examples(self, mode='train'):
        """
        Load data and cache examples
        """
        global enc_input_extend_vocab, title_oovs
        if mode == 'train':
            file = self.args.train_data_path
        elif mode == 'dev':
            file = self.args.dev_data_path
        else:
            raise ValueError(f'Invalid mode: {mode}')

        assert file is not None, f'No files found in {self.args.train_data_path}'

        # If the cache file exists, load it
        cache_path = os.path.join(self.args.software_cache_path, '{}.pkl'.format(mode))
        if os.path.exists(cache_path):
            self.logger.info(f'Loading {mode} data from cache at {self.args.software_cache_path}/{mode}.pkl')
            with open(cache_path, 'rb') as f:
                features = pickle.load(f)
            return features

        # Otherwise, load data from file
        self.logger.info(f"Data cache don't exist, loading {mode} data from {file}")
        with open(file, 'r', encoding='utf-8') as f:
            all_data = json.load(f)

        # pkuseg分词
        seg = pkuseg.pkuseg('web')

        # get id of SOS and EOS, the sign of start and end in decoding
        SOS, EOS = self.vocab.word2id('<SOS>'), self.vocab.word2id('<EOS>')

        # max_len of input and output
        max_enc_len, max_dec_len = self.args.max_enc_len, self.args.max_dec_len

        features = []

        for pair in tqdm(all_data):

            ## Process content
            content = pair['content'].strip()
            content_words = seg.cut(content)
            # truncate if too long
            content_words = content_words[:max_enc_len]

            # list of word ids, OOVs are represented by the id of UNK token
            enc_input = [self.vocab.word2id(word) for word in content_words]
            enc_input_len = len(enc_input)

            ## Process title
            title = pair['title'].strip()
            title_words = seg.cut(title)
            # truncate if too long
            title_words = title_words[:max_dec_len]
            # list of word ids; OOVs are represented by the id for UNK token
            title_ids = [self.vocab.word2id(word) for word in title_words]
            # add SOS and EOS
            dec_input, target = get_dec_input_and_target(title_ids, max_dec_len, SOS, EOS)
            dec_input_len = len(dec_input)

            # if use pointer-generator mode, we need to store some extra information
            enc_input_extend_vocab, content_oovs = None, None
            if self.args.use_pointer_gen:
                # Store a version of the enc_input where in-content OOVs are represented by their temporary content OOV id.
                # also store the in-content OOVs words themselves
                enc_input_extend_vocab, content_oovs = content2ids(content_words, self.vocab)

                # Get a version of the reference title where in-content OOVs are represented by their temporary content OOV id.
                dec_input_extend_vocab = title2ids(title_words, self.vocab, content_oovs)

                # Overwrite decoder target sequence so it uses the temp content OOV ids.
                _, target = get_dec_input_and_target(dec_input_extend_vocab, max_dec_len, SOS, EOS)

            assert len(enc_input) == enc_input_len, f'enc_input_len: {len(enc_input)} != enc_input_len: {enc_input_len}'
            assert len(dec_input) == dec_input_len, f'dec_input_len: {len(dec_input)} != dec_input_len: {dec_input_len}'

            features.append(
                Feature(enc_input=enc_input,
                        enc_input_len=enc_input_len,
                        dec_input=dec_input,
                        dec_input_len=dec_input_len,
                        target=target,
                        title=title,
                        content=content,
                        # if use pointer-generator
                        enc_input_extend_vocab=enc_input_extend_vocab,
                        content_oovs=content_oovs)
            )

        # Save cache
        self.logger.info(f'Saving {mode} data to cache at {self.args.software_cache_path}')
        with open(cache_path, 'wb') as f:
            pickle.dump(features, f)

        return features

if __name__ == '__main__':
    ## Dataloader Test
    from loguru import logger
    from processing.data_utils import Vocab, SummaryDataset

    args = get_parser()
    processor = Processor(args, Vocab(args.vocab_path, args.vocab_size, logger), logger)

    features = processor.load_and_cache_examples('train')

    dataset = SummaryDataset(features)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    print(next(iter(dataloader)))

import csv
import torch
from torch.utils.data import Dataset

# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '<PAD>'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
PAD_ID = 0
max_dec_len = 20
UNKNOWN_TOKEN = '<UNK>'  # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '<SOS>'  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '<EOS>'  # This has a vocab id, which is used at the end of untruncated target sequences


# Note: <s>, </s>, <PAD>, <UNK>, <SOS>, <EOS> shouldn't appear in the vocab file.


class Vocab(object):

    def __init__(self, vocab_file, max_size, logger):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0  # keeps track of total number of words in the Vocab

        # <PAD>, <UNK>, <SOS> and <EOS> get the ids 0,1,2,3.
        for w in [PAD_TOKEN, UNKNOWN_TOKEN, START_DECODING, STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r', encoding='utf-8') as vocab_f:
            for line in vocab_f:
                w = line.strip()

                if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception(
                        '<s>, </s>, <PAD>, <UNK>, <SOS>, <EOS> shouldn\'t be in the vocab file, but %s is' % w)

                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1

                if max_size != 0 and self._count >= max_size:
                    logger.info("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (
                        max_size, self._count))
                    break

        logger.info("Finished constructing vocabulary of %i total words. Last word added: %s" % (
            self._count, self._id_to_word[self._count - 1]))

    def word2id(self, word):
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        return self._count

    def write_metadata(self, fpath):
        print("Writing word embedding metadata file to %s..." % (fpath))
        with open(fpath, "w") as f:
            fieldnames = ['word']
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            for i in range(self.size()):
                writer.writerow({"word": self._id_to_word[i]})


class Feature(object):

    def __init__(self, enc_input, enc_input_len, dec_input, dec_input_len, target,
                 title, content, enc_input_extend_vocab=None, content_oovs=None):
        self.enc_input = enc_input  # id of words in the encoder input sequence
        self.enc_input_len = enc_input_len  # length of the encoder input sequence
        self.dec_input = dec_input  # id of words in the decoder input sequence with <SOS>
        self.dec_input_len = dec_input_len  # length of the decoder input sequence
        self.target = target  # id of words in the target sequence with <EOS>
        self.title = title  # title
        self.content = content  # content

        self.enc_input_extend_vocab = enc_input_extend_vocab  # id of words in the encoder using pointer-generator model
        self.content_oovs = content_oovs  # id of words in the title sequence using pointer-generator model

    def __str__(self):
        return "enc_input: %s\nenc_input_len: %s\ndec_input: %s\ndec_input_len: %s\ntarget: %s\nenc_input_extend_vocab: %s\ntitle_oovs: %s\ntitle: %s\ncontent: %s" % (
            self.enc_input, self.enc_input_len, self.dec_input, self.dec_input_len, self.target,
            self.enc_input_extend_vocab, self.content_oovs, self.title, self.content)


class SummaryDataset(Dataset):
    """
    The dataset of Summary task.
    """

    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

    @staticmethod
    def collate_fn(batch):
        """
        Collate function for data_loader.
        """
        global enc_batch_extend_vocab, content_oovs, extra_zeros

        batch_size = len(batch)

        # Determine the maximum length of the encoder input sequence
        max_enc_len = max([feature.enc_input_len for feature in batch])

        enc_batch = torch.zeros((batch_size, max_enc_len), dtype=torch.long)
        enc_batch_len = torch.zeros(batch_size, dtype=torch.long)
        enc_padding_mask = torch.zeros((batch_size, max_enc_len), dtype=torch.float)

        dec_batch = torch.zeros((batch_size, max_dec_len), dtype=torch.long)
        target_batch = torch.zeros((batch_size, max_dec_len), dtype=torch.long)
        dec_padding_mask = torch.zeros((batch_size, max_dec_len), dtype=torch.float)
        dec_batch_len = torch.zeros(batch_size, dtype=torch.long)

        if batch[0].enc_input_extend_vocab is not None:     # use pointer-generator model
            max_content_oovs = max([len(feature.content_oovs) for feature in batch])
            if max_content_oovs > 0:
                extra_zeros = torch.zeros((batch_size, max_content_oovs), dtype=torch.long)
            content_oovs = [feature.content_oovs for feature in batch]
            enc_batch_extend_vocab = torch.zeros((batch_size, max_enc_len), dtype=torch.long)

        for idx, feature in enumerate(batch):
            # Pad the encoder input sequences up to the length of the longest sequence
            enc_input = feature.enc_input + [PAD_ID] * (max_enc_len - feature.enc_input_len)
            if feature.enc_input_extend_vocab is not None:
                enc_input_extend_vocab = feature.enc_input_extend_vocab + [PAD_ID] * (max_enc_len - feature.enc_input_len)
                enc_batch_extend_vocab[idx, :] = torch.LongTensor(enc_input_extend_vocab)

            enc_batch[idx, :] = torch.LongTensor(enc_input)
            enc_batch_len[idx] = feature.enc_input_len
            enc_padding_mask[idx, :feature.enc_input_len] = 1

            # Pad the decoder input and targets
            dec_input = feature.dec_input + [PAD_ID] * (max_dec_len - feature.dec_input_len)
            target = feature.target + [PAD_ID] * (max_dec_len - feature.dec_input_len)

            dec_batch[idx, :] = torch.LongTensor(dec_input)
            target_batch[idx, :] = torch.LongTensor(target)
            dec_padding_mask[idx, :feature.dec_input_len] = 1
            dec_batch_len[idx] = feature.dec_input_len


        return enc_batch, enc_batch_len, enc_padding_mask, dec_batch, dec_batch_len, dec_padding_mask,\
               target_batch, enc_batch_extend_vocab, extra_zeros, content_oovs, \
               [feature.title for feature in batch], [feature.content for feature in batch]

        # return {
        #     'enc_input': enc_batch,
        #     'enc_input_len': enc_batch_len,
        #     'enc_padding_mask': enc_padding_mask,
        #     'dec_input': dec_batch,
        #     'dec_input_len': dec_batch_len,
        #     'dec_padding_mask': dec_padding_mask,
        #     'target': target_batch,
        #     'content_oovs': content_oovs,
        #     'enc_input_extend_vocab': enc_batch_extend_vocab,
        #
        #     'title': [feature.title for feature in batch],
        #     'content': [feature.content for feature in batch]
        # }


if __name__ == '__main__':
    pass



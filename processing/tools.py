import random
import os
import torch
import numpy as np


def seed_everything(seed=1234):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def save_model(model, logger, path):
    torch.save(model.state_dict(), path)
    logger.info(f'Saved model to {path}')


def get_dec_input_and_target(title_ids, max_dec_len, SOS, EOS):
    """
    Args:
        title_ids: List of title(ids), OOVs are represented by the id of UNK token
        max_dec_len: The max length in decoding process
        SOS: Id of start sign
        EOS: Id of end sign

    :return: decoder_input, target
    """
    input = [SOS] + title_ids[:]
    target = title_ids[:]

    if len(input) > max_dec_len:
        input = input[:max_dec_len]
        target = target[:max_dec_len]   # has no end token
    else:
        target.append(EOS)  # has end token

    assert len(input) == len(target)

    return input, target


def title2ids(title_words, vocab, content_oovs):
    """
    Args:
        title_words: List of title(words), OOVs are represented by the word of UNK token
        vocab: Vocabulary object
        content_oov: OOV of content
    :return: title_ids: List of title(ids)
    """
    ids = []
    unk_id = vocab.word2id('<UNK>')

    for word in title_words:
        id = vocab.word2id(word)
        if id == unk_id:    # If word is an OOV
            if word in content_oovs:  # If w is an in-title OOV
                vocab_idx = vocab.size() + content_oovs.index(word)
                ids.append(vocab_idx)
            else:   # OOV but not in in-title OOVs
                ids.append(unk_id)
        else:   # Not OOV
            ids.append(id)

    return ids


def content2ids(content_words, vocab):
    """
    Args:
        content_words: List of content(words), OOVs are represented by the word of UNK token
        vocab: Vocabulary object
    :return: content_ids: List of content(ids), oovs: OOVs in this content
    """
    ids = []
    oovs = []
    unk_id = vocab.word2id('<UNK>')

    for word in content_words:
        id = vocab.word2id(word)
        if id == unk_id:  # If word is an OOV
            if word not in oovs:  # Add to list of OOVs
                oovs.append(word)
            oov_id = oovs.index(word)  # Get its id, 0 for first OOV in title, 1 for second OOV, ...
            ids.append(vocab.size() + oov_id)
        else:  # Not OOV
            ids.append(id)

    return ids, oovs

def outputids2words(id_list, vocab, article_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i)  # might be [UNK]
        except ValueError as e:  # w is OOV
            assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e:  # i doesn't correspond to an article oov
                raise ValueError(
                    'Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (
                    i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words

def decode_output_ids(output_ids, vocab, content_oovs):
    if output_ids < vocab.size():
        return vocab.id2word(output_ids)
    else:
        return content_oovs[output_ids - vocab.size()]


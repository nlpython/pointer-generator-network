import argparse
import six

def print_arguments(args, log):
    log.info('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        log.info('%s: %s' % (arg, value))
    log.info('------------------------------------------------')

def get_parser():

    parser = argparse.ArgumentParser(description='Hyperparameters')

    # data settings
    parser.add_argument('--train_data_path', type=str, default='./software_data/train.json',
                        help='path of train data')
    parser.add_argument('--dev_data_path', type=str, default='./software_data/dev.json',
                        help='path of dev data')
    parser.add_argument('--software_cache_path', type=str, default='./software_data/cache',
                        help='path of data')
    parser.add_argument('--vocab_path', type=str, default='./software_data/vocab',
                        help='path of vocab file')
    parser.add_argument('--vocab_size', type=int, default=50000,
                        help='max vocab size of all vocabularies')


    # training settings
    parser.add_argument('--log_path', type=str, default='./logs',
                        help='path of log file')
    parser.add_argument('--max_enc_len', type=int, default=800,
                        help='max length of encoder')
    parser.add_argument('--max_dec_len', type=int, default=20,  # make sure equal to max_enc_len in file(data_utils.py)
                        help='max length of decoder')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size of train and evaluate')
    parser.add_argument('--lr', type=float, default=0.15,
                        help='learning rate of other layers')
    parser.add_argument('--lr_coverage', type=float, default=0.15,
                        help='learning rate of coverage layer')
    parser.add_argument('--adagrad_init_acc', type=float, default=0.1,
                        help='initial accumulator value for Adagrad')
    parser.add_argument('--epochs', type=int, default=35,
                        help='number of epochs')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay of all layers')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--warmup_proportion', type=float, default=0.1,
                        help='proportion of training steps to perform linear learning rate warmup for')
    parser.add_argument('--adam_epsilon', type=float, default=1e-12,
                        help='epsilon of adam')
    parser.add_argument('--max_clip_norm', type=float, default=2.0,
                        help='max norm of gradients')

    parser.add_argument('--use_pointer_gen', type=bool, default=True,
                        help='if use pointer generator')

    parser.add_argument('--beam_size', type=int, default=2,
                        help='beam size for beam search')


    # model settings
    parser.add_argument('--use_coverage', type=bool, default=False,
                        help='use coverage or not')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='embedding dimension of LstmClassifier')
    parser.add_argument('--embedding_init_std', type=float, default=1e-4,
                        help='range of normal distribution to initialize embedding')
    parser.add_argument('--hidden_dim', type=int, default=256,              # todo *2
                        help='hidden dimension of LstmClassifier')
    parser.add_argument('--lstm_init_range', type=float, default=0.02,
                        help='range of uniform distribution to initialize lstm')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout rate')
    parser.add_argument('--source_max_length', type=int, default=512,
                        help='max length of source sentence')
    parser.add_argument('--target_max_length', type=int, default=128,
                        help='max length of absract sentence')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints',
                        help='path of checkpoint file')


    return parser.parse_args()

if __name__ == '__main__':
    get_parser()


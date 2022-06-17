from processing.hyper_parm import print_arguments, get_parser
import torch
import time
from loguru import logger
from processing.data_utils import Vocab, SummaryDataset
from processing.processor import Processor
from processing.tools import save_model, seed_everything
from torch.utils.data import DataLoader
from model.model import SummaryModel


def train():

    # define args of training and model
    args = get_parser()

    # If use cuda, set cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device

    # seed everything
    seed_everything(args.seed)
    # define logger
    logger.add('./logs/{time}.log')
    print_arguments(args, logger)

    # define vocab
    vocab = Vocab(args.vocab_path, args.vocab_size, logger)
    args.vocab_size = vocab.size()

    # define processor
    processor = Processor(args, vocab, logger)

    # build dataloader
    train_dataset = SummaryDataset(processor.load_and_cache_examples('train'))
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )

    dev_dataset = SummaryDataset(processor.load_and_cache_examples('dev'))
    dev_dataloader = DataLoader(
        dataset=dev_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dev_dataset.collate_fn
    )

    # define model
    model = SummaryModel(args).to(device)

    # define optimizer
    parameters = list(model.encoder.parameters()) + list(model.decoder.parameters()) + \
                 list(model.reduce_state.parameters())

    initial_lr = args.lr_coverage if args.use_coverage else args.lr
    optimizer = torch.optim.Adagrad(parameters, lr=initial_lr, initial_accumulator_value=args.adagrad_init_acc)
    # optimizer = torch.optim.Adam(parameters, lr=initial_lr)

    logger.info('\n')
    logger.info("***** Running training *****")
    logger.info("  Num Epochs = {}".format(args.epochs))
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = {}".format(
        args.batch_size))
    logger.info("  Type of optimizer = {}".format(optimizer.__class__.__name__))
    logger.info("  Total optimization steps = {}".format(len(train_dataloader) * args.epochs))
    logger.info("  Learning rate = {}".format(args.lr))
    logger.info('\n')

    coverage = None
    for epoch in range(args.epochs):
        model.train()
        running_avg_loss, batch_num = 0.0, 0
        for step, batch in enumerate(train_dataloader):
            model.zero_grad()
            batch = tuple(t.to(device) for t in batch[:-3] if t is not None) + batch[-3:]
            enc_batch, enc_batch_len, enc_padding_mask, dec_batch, dec_batch_len, dec_padding_mask, \
            target_batch, enc_batch_extend_vocab, extra_zeros, content_oovs, title, content = batch

            # forward
            c_t_1 = torch.zeros((enc_batch.shape[0], 2*args.hidden_dim), device=device)

            if args.use_coverage:
                coverage = torch.zeros((enc_batch.shape[0]), device=device)

            # [B*2, L, H], [B*L, H*2], ([2, B, H]*2)
            encoders_outputs, encoder_features, encoder_hidden = \
                model.encoder(enc_batch, enc_batch_len)

            s_t_1 = model.reduce_state(encoder_hidden)  # (h_t, c_t) [1, B, H]

            step_losses = []

            # decode
            for di in range(args.max_dec_len):
                y_t_1 = dec_batch[:, di]  # Teacher forcing [B,]
                final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = \
                    model.decoder(y_t_1, s_t_1, encoders_outputs, encoder_features,
                                  enc_padding_mask, c_t_1, extra_zeros, enc_batch_extend_vocab,
                                  coverage, di)

                target = target_batch[:, di]    # [B,]
                gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze(1)    # [16,]
                step_loss = -torch.log(gold_probs + 1e-10)

                if args.use_coverage:
                    step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                    step_loss = step_loss + args.cov_loss_wt * step_coverage_loss
                    coverage = next_coverage

                step_mask = dec_padding_mask[:, di]
                step_loss = step_loss * step_mask   # [B,]
                step_losses.append(step_loss)

            # compute loss of this batch
            total_loss = torch.sum(torch.stack(step_losses, 1), 1)
            batch_avg_loss = total_loss / dec_batch_len
            loss = torch.mean(batch_avg_loss)

            # compute avg loss
            running_avg_loss += loss.item()
            batch_num += 1

            loss.backward()

            if args.max_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters, args.max_clip_norm)

            optimizer.step()

            if step % 50 == 0:
                logger.info('Epoch: {:2d} | step: {:4d} | loss: {:6f}'.format(
                    epoch, step, running_avg_loss / batch_num))
                running_avg_loss, batch_num = 0.0, 0

        # save model
        torch.save(model.state_dict(), './checkpoints/{}_{}.pth'.format(model.__class__.__name__, epoch))
        logger.info('Successfully saved model at epoch {}'.format(epoch))




if __name__ == '__main__':
    train()


    # args = get_parser()
    # processor = Processor(args, Vocab(args.vocab_path, args.vocab_size, logger), logger)
    #
    # features = processor.load_and_cache_examples('train')
    #
    # dataset = SummaryDataset(features)
    #
    # from torch.utils.data import DataLoader
    #
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    # print(next(iter(dataloader)))


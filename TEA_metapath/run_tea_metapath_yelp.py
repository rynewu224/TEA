import argparse
import os
from datetime import datetime
from time import time

import torch.multiprocessing

from models import *
from utils import *


def parse_sampled_batch(batch, seq_maxlen, nbr_maxlen, metapath_maxlen):
    uid, seq, pos, neg, nbr, nbr_iid, metapath = batch
    uid = uid.long()
    seq = seq.long()  # [:, :seq_maxlen]
    pos = pos.long()  # [:, :seq_maxlen]
    neg = neg.long()  # [:, :seq_maxlen]
    nbr = nbr.long()  # [:, :nbr_maxlen]
    nbr_iid = nbr_iid.long()  # [:, :nbr_maxlen, :seq_maxlen]
    metapath = metapath.long()[:, :seq_maxlen, :metapath_maxlen]
    batch = [uid, seq, pos, neg, nbr, nbr_iid, metapath]
    indices = torch.where(pos != 0)
    return batch, indices


def train(model, opt, lr_scheduler, train_loader, args):
    seq_maxlen, nbr_maxlen, metapath_maxlen = args.seq_maxlen, args.nbr_maxlen, args.metapath_maxlen
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        parsed_batch, indices = parse_sampled_batch(batch, seq_maxlen, nbr_maxlen, metapath_maxlen)
        opt.zero_grad()
        pos_logits, neg_logits, user_emb, pos_item_emb, neg_item_emb = model.dual_forward(parsed_batch)

        loss = 0.0
        if args.loss_type == 'bce':
            loss += F.binary_cross_entropy_with_logits(pos_logits[indices], torch.ones_like(pos_logits)[indices]) + \
                    F.binary_cross_entropy_with_logits(neg_logits[indices], torch.zeros_like(neg_logits)[indices])
        elif args.loss_type == 'bpr':
            loss += F.softplus(neg_logits[indices] - pos_logits[indices]).mean()
        elif args.loss_type == 'sfm':
            uid, seq, pos, neg, nbr, nbr_iid, meta = parsed_batch
            all_items = torch.cat([pos.unsqueeze(-1), neg], dim=-1)  # B x sl x (1 + ns)
            all_indices = torch.where(all_items != 0)
            logits = torch.cat([pos_logits, neg_logits], dim=-1)  # B x sl x (1 + ns)
            logits = logits[all_indices].view(-1, 1 + args.neg_size)
            device = torch.device(f'{args.device}')
            labels = torch.zeros((logits.shape[0])).long().to(device)
            loss += F.cross_entropy(logits, labels)

        # Embedding Reg loss
        user_norm = user_emb.norm(2, dim=-1).pow(2).mean()
        item_norm = pos_item_emb.norm(2, dim=-1).pow(2).mean() + neg_item_emb.norm(2, dim=-1).pow(2).mean()
        emb_reg_loss = args.emb_reg * 0.5 * (user_norm + item_norm)
        loss += emb_reg_loss

        loss.backward()
        opt.step()
        lr_scheduler.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def main():
    parser = argparse.ArgumentParser(description='TEA_metapath_v1')
    parser.add_argument('--dataset', default='Epinions')
    parser.add_argument('--model', default='TEA_metapath_v1')

    # Model Config
    parser.add_argument('--edim', type=int, default=64)
    parser.add_argument('--seq_maxlen', type=int, default=50, help='no larger than 50')
    parser.add_argument('--nbr_maxlen', type=int, default=20, help='no larger than 20')
    parser.add_argument('--metapath_maxlen', type=int, default=10, help='no larger than 10')
    parser.add_argument('--neg_size', type=int, default=50, help='Negative samples number')
    parser.add_argument('--aggr_type', type=str, default='sage',
                        help='GNN type of social aggregation, sage(GraphSAGE) or gat(GAT)')

    # Train Config
    parser.add_argument('--batch_size', type=int, default=1024, help='fixed, or change with sampled train_batches')
    parser.add_argument('--droprate', type=float, default=0.5)  # 0.5
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_gamma', type=float, default=0.001)
    parser.add_argument('--lr_decay_rate', type=float, default=0.75)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--emb_reg', type=float, default=5e-4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--check_per_epoch', type=int, default=1)
    parser.add_argument('--check_start_epoch', type=int, default=0)
    parser.add_argument('--loss_type', default='sfm', help='bce/bpr/sfm')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--patience', type=int, default=5)

    # Something else
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--test_time', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=False)
    args = parser.parse_args()

    user_num, item_num = np.load(f'datasets/{args.dataset}/user_item_num.npy')
    user_num = user_num + 1
    item_num = item_num + 1

    print('Loading...')
    st = time()
    train_loader, val_loader, test_loader, user_train, eval_users = load_ds(args, item_num)
    print('Loaded {} dataset with {} users {} items in {:.2f}s'.format(args.dataset, user_num, item_num, time() - st))
    timestr = datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss")
    model_path = f'saved_models/{args.model}_{args.dataset}_{timestr}.pth'
    logger = get_logger(os.path.join('logs', f'{args.model}_{args.dataset}_{timestr}.log'))
    logger.info(args)
    device = torch.device(args.device)

    metrics_list = []
    for r in range(args.repeat):
        seed = args.seed + r
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        model = TEA_metapath_v1(user_num, item_num, args).to(device)
        opt = torch.optim.Adam(model.get_parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = StepwiseLR(opt, init_lr=args.lr, gamma=args.lr_gamma, decay_rate=args.lr_decay_rate)
        best_score = patience_cnt = 0
        for epoch in range(1, args.max_epochs):
            st = time()
            train_loss = train(model, opt, lr_scheduler, train_loader, args)
            print('Epoch:{} Train Loss={:.4f} Time={:.2f}s LR={:.8f}'.format(
                epoch, train_loss, time() - st, lr_scheduler.get_lr()))

            if epoch % args.check_per_epoch == 0 and epoch >= args.check_start_epoch:
                val_metrics = evaluate(model, val_loader, eval_users)
                hr5, hr10, hr20, ndcg5, ndcg10, ndcg20 = val_metrics
                logger.info(
                    'Iter={} Epoch={:04d} Val HR(5/10/20)={:.4f}/{:.4f}/{:.4f} NDCG(5/10/20)={:.4f}/{:.4f}/{:.4f}'
                        .format(r, epoch, hr5, hr10, hr20, ndcg5, ndcg10, ndcg20))

                if best_score < hr10:
                    torch.save(model.state_dict(), model_path)
                    print('Validation HitRate@10 increased: {:.4f} --> {:.4f}'.format(best_score, hr10))
                    best_score = hr10
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    print(f'Patience {patience_cnt}/{args.patience}')

                if patience_cnt == args.patience:
                    print('Early Stop!!!')
                    break

        print('Testing')
        model.load_state_dict(torch.load(model_path))
        test_metrics = evaluate(model, test_loader, eval_users)
        hr5, hr10, hr20, ndcg5, ndcg10, ndcg20 = test_metrics
        logger.info('Iter={} Tst HR(5/10/20)={:.4f}/{:.4f}/{:.4f} NDCG(5/10/20)={:.4f}/{:.4f}/{:.4f}'
                    .format(r, hr5, hr10, hr20, ndcg5, ndcg10, ndcg20))
        metrics_list.append(test_metrics)

    metrics = np.array(metrics_list)
    means = metrics.mean(axis=0)
    stds = metrics.std(axis=0)
    print(f'{args.model} {args.dataset} Test Summary:')
    logger.info('Mean hr5={:.4f}, hr10={:.4f}, hr20={:.4f}, ndcg5={:.4f}, ndcg10={:.4f}, ndcg20={:.4f}'.format(
        means[0], means[1], means[2], means[3], means[4], means[5]))
    logger.info('Std  hr5={:.4f}, hr10={:.4f}, hr20={:.4f}, ndcg5={:.4f}, ndcg10={:.4f}, ndcg20={:.4f}'.format(
        stds[0], stds[1], stds[2], stds[3], stds[4], stds[5]))
    logger.info("Done")


if __name__ == '__main__':
    main()

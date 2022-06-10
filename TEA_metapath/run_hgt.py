import argparse
import os
from datetime import datetime
from time import time

import torch.multiprocessing

from models import *
from utils import *


def parse_sampled_batch(batch):
    uid, seq, pos, neg, nbr, nbr_iid, metapath = batch
    uid = uid.long()
    pos = pos.long()
    neg = neg.long()
    batch = [uid, pos, neg]
    indices = torch.where(pos != 0)
    return batch, indices


def train(model, opt, train_loader, args):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        parsed_batch, indices = parse_sampled_batch(batch)
        opt.zero_grad()
        pos_logits, neg_logits, user_emb, pos_item_emb, neg_item_emb = model(parsed_batch)

        loss = 0.0
        if args.loss_type == 'bce':
            loss += F.binary_cross_entropy_with_logits(pos_logits[indices], torch.ones_like(pos_logits)[indices]) + \
                    F.binary_cross_entropy_with_logits(neg_logits[indices], torch.zeros_like(neg_logits)[indices])
        elif args.loss_type == 'bpr':
            loss += F.softplus(neg_logits[indices] - pos_logits[indices]).mean()
        elif args.loss_type == 'sfm':
            uid, pos, neg = parsed_batch
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
        total_loss += loss.item()

    return total_loss / len(train_loader)


def main():
    parser = argparse.ArgumentParser(description='HGT')
    parser.add_argument('--dataset', default='Wechat')
    parser.add_argument('--model', default='HGT')

    # Model Config
    parser.add_argument('--edim', type=int, default=32)
    parser.add_argument('--seq_maxlen', type=int, default=50, help='no larger than 50')
    parser.add_argument('--nbr_maxlen', type=int, default=20, help='no larger than 20')
    parser.add_argument('--metapath_maxlen', type=int, default=10, help='no larger than 10')
    parser.add_argument('--neg_size', type=int, default=10, help='Negative samples number')

    # Train Config
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--emb_reg', type=float, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--check_per_epoch', type=int, default=1)
    parser.add_argument('--check_start_epoch', type=int, default=0)
    parser.add_argument('--loss_type', default='sfm', help='bce/bpr/sfm')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--patience', type=int, default=3)

    # Something else
    parser.add_argument('--repeat', type=int, default=2)
    parser.add_argument('--test_time', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=False)
    args = parser.parse_args()

    print('Loading...')
    st = time()
    user_num, item_num = 1 + np.load(f'datasets/{args.dataset}/user_item_num.npy')
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

        G, node_dict, edge_dict = build_dgl_graph(args, user_num, item_num, user_train)
        G = G.to(device)
        model = HGTRec(G, node_dict, edge_dict,
                       args.edim, args.edim, args.edim,
                       args.num_layers, args.num_heads).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

        best_score = patience_cnt = 0
        for epoch in range(1, args.max_epochs+1):
            st = time()
            train_loss = train(model, opt, train_loader, args)
            print('Epoch:{} Train Loss={:.4f} Time={:.2f}s'.format(epoch, train_loss, time() - st))
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
    logger.info('Mean hr5={:.4f}, ndcg5={:.4f}, hr10={:.4f}, ndcg10={:.4f}, hr20={:.4f}, ndcg20={:.4f}'.format(
        means[0], means[3], means[1], means[4], means[2], means[5]))
    logger.info('Std  hr5={:.4f}, ndcg5={:.4f}, hr10={:.4f}, ndcg10={:.4f}, hr20={:.4f}, ndcg20={:.4f}'.format(
        stds[0], stds[3], stds[1], stds[4], stds[2], stds[5]))

    logger.info("Done")


if __name__ == '__main__':
    main()


import torch
import numpy as np
from utils import all_metrics, print_metrics


def train(args, mlb, model, optimizer, epoch, gpu, data_loader, G):

    print("EPOCH %d" % epoch)

    losses = []


    model.train()

    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in range(num_iter):

        if args.model.find("bert") != -1:

            inputs_id, segments, masks, labels = next(data_iter)

            inputs_id, segments, masks, labels = torch.LongTensor(inputs_id), torch.LongTensor(segments), \
                                                 torch.LongTensor(masks), torch.FloatTensor(labels)

            if gpu >= 0:
                inputs_id, segments, masks, labels = inputs_id.cuda(gpu), segments.cuda(gpu), \
                                                     masks.cuda(gpu), labels.cuda(gpu)

            output, loss = model(inputs_id, segments, masks, labels)
        else:

            inputs_id, seq_len, labels, masks = next(data_iter)

            inputs_id = torch.LongTensor(inputs_id)
            seq_len = torch.Tensor(seq_len)
            labels = torch.from_numpy(mlb.fit_transform(labels)).type(torch.float)
            masks = torch.from_numpy(mlb.fit_transform(masks)).type(torch.float)

            if gpu >= 0:
                inputs_id, labels, masks = inputs_id.cuda(gpu), labels.cuda(gpu), masks.cuda(gpu)
                G, G.ndata['feat'] = G.to('cuda'), G.ndata['feat'].to('cuda')

            output, loss = model(inputs_id, seq_len, labels)
            # output, loss = model(inputs_id, labels, masks, G, G.ndata['feat'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses


def test(args, mlb, model, data_path, fold, gpu, dicts, data_loader, G):

    filename = data_path.replace('train', fold)
    print('file for evaluation: %s' % filename)
    num_labels = len(dicts['ind2c'])

    y, yhat, yhat_raw, hids, losses = [], [], [], [], []

    model.eval()

    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in range(num_iter):
        with torch.no_grad():

            if args.model.find("bert") != -1:
                inputs_id, segments, masks, labels = next(data_iter)

                inputs_id, segments, masks, labels = torch.LongTensor(inputs_id), torch.LongTensor(segments), \
                                                     torch.LongTensor(masks), torch.FloatTensor(labels)

                if gpu >= 0:
                    inputs_id, segments, masks, labels = inputs_id.cuda(
                        gpu), segments.cuda(gpu), masks.cuda(gpu), labels.cuda(gpu)

                output, loss = model(inputs_id, segments, masks, labels)
            else:

                inputs_id, seq_len, labels, masks = next(data_iter)

                inputs_id = torch.LongTensor(inputs_id)
                seq_len = torch.Tensor(seq_len)
                labels = torch.from_numpy(mlb.fit_transform(labels)).type(torch.float)
                masks = torch.from_numpy(mlb.fit_transform(masks)).type(torch.float)

                if gpu >= 0:
                    inputs_id, labels, masks = inputs_id.cuda(gpu), labels.cuda(gpu), masks.cuda(gpu)
                    G, G.ndata['feat'] = G.to('cuda'), G.ndata['feat'].to('cuda')

                output, loss = model(inputs_id, seq_len, labels)
                # output, loss = model(inputs_id, labels, masks, G, G.ndata['feat'])

            output = torch.sigmoid(output)
            output = output.data.cpu().numpy()

            losses.append(loss.item())
            target_data = labels.data.cpu().numpy()

            yhat_raw.append(output)
            output = np.round(output)
            y.append(target_data)
            yhat.append(output)

    y = np.concatenate(y, axis=0)
    yhat = np.concatenate(yhat, axis=0)
    yhat_raw = np.concatenate(yhat_raw, axis=0)

    k = 5 if num_labels == 50 else [8, 15]
    metrics = all_metrics(yhat, y, k=k, yhat_raw=yhat_raw)
    print_metrics(metrics)
    metrics['loss_%s' % fold] = np.mean(losses)
    return metrics

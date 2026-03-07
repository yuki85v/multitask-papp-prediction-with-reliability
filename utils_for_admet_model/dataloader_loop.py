import numpy as np
import torch


def train_loop(
        encoder, decoder, dataloader, loss_func, optimizer, device,
        scheduler, epoch, warmup_epochs
    ):

    encoder.train()
    decoder.train()
    losses = 0

    if scheduler is not None:
        if epoch < warmup_epochs:
            for param_group in optimizer.param_groups:
                init_lr = param_group['initial_lr']
                param_group['lr'] = init_lr * (epoch + 1) / warmup_epochs
        else:
            scheduler.step()

    for data in dataloader:
        _, graphs, labels = data
        graphs = graphs.to(device)
        labels = labels.to(device)

        graph_feats = encoder(graphs, graphs.ndata['h'])
        preds = decoder(graph_feats)
        loss = loss_func(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss.item() * labels.shape[0]

    losses /= len(dataloader.dataset)

    return losses


def eval_loop(encoder, decoder, dataloader, loss_func, device):
    encoder.eval()
    decoder.eval()
    losses = 0
    outputs = {
        'compids': [],
        'labels': [],
        'preds': [],
    }

    with torch.no_grad():
        for data in dataloader:
            compids, graphs, labels = data
            graphs = graphs.to(device)
            labels = labels.to(device)

            graph_feats = encoder(graphs, graphs.ndata['h'])
            preds = decoder(graph_feats)
            loss = loss_func(preds, labels)

            losses += loss.item() * labels.shape[0]

            outputs['compids'].extend(compids)
            outputs['labels'].extend(labels.cpu().detach().numpy())
            outputs['preds'].extend(preds.cpu().detach().numpy())

    outputs['compids'] = np.array(outputs['compids'])
    outputs['labels'] = np.vstack(outputs['labels'])
    outputs['preds'] = np.vstack(outputs['preds'])

    losses /= len(dataloader.dataset)

    return losses, outputs


def infer_loop(encoder, decoder, dataloader, device):
    encoder.eval()
    decoder.eval()
    outputs = {
        'compids': [],
        'preds': [],
        'graph_feats': []
    }

    with torch.no_grad():
        for data in dataloader:
            compids, graphs, _ = data
            graphs = graphs.to(device)

            graph_feats = encoder(graphs, graphs.ndata['h'])
            preds = decoder(graph_feats)

            outputs['compids'].extend(compids)
            outputs['preds'].extend(preds.cpu().detach().numpy())
            outputs['graph_feats'].extend(graph_feats.cpu().detach().numpy())

    outputs['compids'] = np.array(outputs['compids'])
    outputs['preds'] = np.vstack(outputs['preds'])
    outputs['graph_feats'] = np.vstack(outputs['graph_feats'])

    return outputs

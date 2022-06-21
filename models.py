from math import floor

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as xavier_uniform
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import build_pretrain_embedding, load_embeddings

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        """
        inputs: g,       object of Graph
                feature, node features
        """
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)


class LabelNet(nn.Module):
    def __init__(self, hidden_gcn_size, num_classes, in_node_features):
        super(LabelNet, self).__init__()
        self.gcn1 = GCNLayer(in_node_features, hidden_gcn_size)
        self.gcn2 = GCNLayer(hidden_gcn_size, num_classes)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = F.relu(x)
        x = self.gcn2(g, x)
        return x


class CorNetBlock(nn.Module):
    def __init__(self, context_size, output_size):
        super(CorNetBlock, self).__init__()
        self.dstbn2cntxt = nn.Linear(output_size, context_size)
        self.cntxt2dstbn = nn.Linear(context_size, output_size)
        self.act_fn = torch.sigmoid

    def forward(self, output_dstrbtn):
        identity_logits = output_dstrbtn
        output_dstrbtn = self.act_fn(output_dstrbtn)
        context_vector = self.dstbn2cntxt(output_dstrbtn)
        context_vector = F.elu(context_vector)
        output_dstrbtn = self.cntxt2dstbn(context_vector)
        output_dstrbtn = output_dstrbtn + identity_logits
        return output_dstrbtn


class CorNet(nn.Module):
    def __init__(self, output_size, cornet_dim=1000, n_cornet_blocks=2):
        super(CorNet, self).__init__()
        self.intlv_layers = nn.ModuleList(
            [CorNetBlock(cornet_dim, output_size) for _ in range(n_cornet_blocks)])
        for layer in self.intlv_layers:
            nn.init.xavier_uniform_(layer.dstbn2cntxt.weight)
            nn.init.xavier_uniform_(layer.cntxt2dstbn.weight)

    def forward(self, logits):
        for layer in self.intlv_layers:
            logits = layer(logits)
        return logits


class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=20):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1)
        return x * y.expand_as(x)


class WordRep(nn.Module):
    def __init__(self, args, Y, dicts):
        super(WordRep, self).__init__()

        self.gpu = args.gpu

        if args.embed_file:
            print("loading pretrained embeddings from {}".format(args.embed_file))
            if args.use_ext_emb:
                pretrain_word_embedding, pretrain_emb_dim = build_pretrain_embedding(args.embed_file, dicts['w2ind'],
                                                                                     True)
                W = torch.from_numpy(pretrain_word_embedding)
            else:
                W = torch.Tensor(load_embeddings(args.embed_file))

            self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.embed.weight.data = W.clone()
        else:
            # add 2 to include UNK and PAD
            self.embed = nn.Embedding(len(dicts['w2ind']) + 2, args.embed_size, padding_idx=0)
        self.feature_size = self.embed.embedding_dim

        self.use_elmo = args.use_elmo
        # if self.use_elmo:
        #     self.elmo = Elmo(args.elmo_options_file, args.elmo_weight_file, 1, requires_grad=args.elmo_tune,
        #                      dropout=args.elmo_dropout, gamma=args.elmo_gamma)
        #     with open(args.elmo_options_file, 'r') as fin:
        #         _options = json.load(fin)
        #     self.feature_size += _options['lstm']['projection_dim'] * 2

        self.embed_drop = nn.Dropout(p=args.dropout)

        self.conv_dict = {1: [self.feature_size, args.num_filter_maps],
                     2: [self.feature_size, 100, args.num_filter_maps],
                     3: [self.feature_size, 150, 100, args.num_filter_maps],
                     4: [self.feature_size, 200, 150, 100, args.num_filter_maps]
                     }

    def forward(self, x, target):

        features = [self.embed(x)]

        # if self.use_elmo:
        #     elmo_outputs = self.elmo(text_inputs)
        #     elmo_outputs = elmo_outputs['elmo_representations'][0]
        #     features.append(elmo_outputs)

        x = torch.cat(features, dim=2)

        x = self.embed_drop(x)
        return x


class OutputLayer(nn.Module):
    def __init__(self, args, Y, dicts, input_size):
        super(OutputLayer, self).__init__()

        self.U = nn.Linear(input_size, Y)
        xavier_uniform(self.U.weight)

        self.final = nn.Linear(input_size, Y)
        xavier_uniform(self.final.weight)

        self.loss_function = nn.BCEWithLogitsLoss()

    def forward(self, x, target, mask):

        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
        # print('alpha', alpha.size())
        m = alpha.matmul(x) # [bs, Y, dim]
        # print('m', m.size())
        # print('mask', mask.size())

        m = m.transpose(1, 2) * mask.unsqueeze(1)
        m = m.transpose(1, 2)
        # print('m', m.size())
        # alpha = torch.softmax(torch.matmul(x, mask), dim=1)
        # m = torch.matmul(x.transpose(1, 2), alpha).transpose(1, 2)   # size: (bs, num_label, 50 * filter_num)

        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        loss = self.loss_function(y, target)
        return y, loss
        # return y


def label_smoothing(y, alpha, Y):
    return y*(1-alpha) + alpha/Y


class OutputLayer_label_smooth(nn.Module):
    def __init__(self, args, Y, dicts, input_size):
        super(OutputLayer_label_smooth, self).__init__()
        self.args = args
        self.Y = Y
        # self.U = nn.Linear(input_size, Y)
        self.final = nn.Linear(input_size, Y)
        # xavier_uniform(self.U.weight)
        xavier_uniform(self.final.weight)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, x, target, mask):
        # att = self.U.weight.matmul(x.transpose(1, 2)) # [bs, Y, seq_len]
        # alpha = F.softmax(att, dim=2)
        # m = alpha.matmul(x)     # [bs, Y, dim]

        alpha = torch.softmax(torch.matmul(x, mask), dim=1)
        m = torch.matmul(x.transpose(1, 2), alpha).transpose(1, 2)   # size: (bs, num_label, 50 * filter_num)

        logits = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        if self.args.label_smoothing:
            target = label_smoothing(target, self.args.alpha, self.Y)
            yhat = torch.sigmoid(logits)
            loss = torch.mean(-target*torch.log(yhat) - (1-target)*torch.log(1-yhat))
        else:
            loss = self.loss_func(logits, target)
        return logits, loss


class CNN(nn.Module):
    def __init__(self, args, Y, dicts):
        super(CNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        filter_size = int(args.filter_size)

        self.conv = nn.Conv1d(self.word_rep.feature_size, args.num_filter_maps, kernel_size=filter_size,
                                  padding=int(floor(filter_size / 2)))
        xavier_uniform(self.conv.weight)

        self.output_layer = OutputLayer(args, Y, dicts, args.num_filter_maps)

    def forward(self, x, target, text_inputs):

        x = self.word_rep(x, target, text_inputs)

        x = x.transpose(1, 2)

        x = torch.tanh(self.conv(x).transpose(1, 2))

        y, loss = self.output_layer(x, target, text_inputs)
        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class MultiCNN(nn.Module):
    def __init__(self, args, Y, dicts):
        super(MultiCNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        if args.filter_size.find(',') == -1:
            self.filter_num = 1
            filter_size = int(args.filter_size)
            self.conv = nn.Conv1d(self.word_rep.feature_size, args.num_filter_maps, kernel_size=filter_size,
                                  padding=int(floor(filter_size / 2)))
            xavier_uniform(self.conv.weight)
        else:
            filter_sizes = args.filter_size.split(',')
            self.filter_num = len(filter_sizes)
            self.conv = nn.ModuleList()
            for filter_size in filter_sizes:
                filter_size = int(filter_size)
                tmp = nn.Conv1d(self.word_rep.feature_size, args.num_filter_maps, kernel_size=filter_size,
                                      padding=int(floor(filter_size / 2)))
                xavier_uniform(tmp.weight)
                self.conv.add_module('conv-{}'.format(filter_size), tmp)

        self.output_layer = OutputLayer(args, Y, dicts, self.filter_num * args.num_filter_maps)

    def forward(self, x, target, text_inputs):

        x = self.word_rep(x, target, text_inputs)

        x = x.transpose(1, 2)

        if self.filter_num == 1:
            x = torch.tanh(self.conv(x).transpose(1, 2))
        else:
            conv_result = []
            for tmp in self.conv:
                conv_result.append(torch.tanh(tmp(x).transpose(1, 2)))
            x = torch.cat(conv_result, dim=2)

        y, loss = self.output_layer(x, target, text_inputs)

        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, use_res, dropout):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel),
            nn.Tanh(),
            nn.Conv1d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel)
        )
        # self.se = SE_Block(outchannel)
        self.use_res = use_res
        if self.use_res:
            self.shortcut = nn.Sequential(
                        nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm1d(outchannel)
                    )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.left(x)
        # out = self.se(out)
        if self.use_res:
            out += self.shortcut(x)
        out = torch.tanh(out)
        out = self.dropout(out)
        return out


class ResCNN(nn.Module):

    def __init__(self, args, Y, dicts):
        super(ResCNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        self.conv = nn.ModuleList()
        conv_dimension = self.word_rep.conv_dict[args.conv_layer]
        for idx in range(args.conv_layer):
            tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], int(args.filter_size), 1, True, args.dropout)
            self.conv.add_module('conv-{}'.format(idx), tmp)

        self.output_layer = OutputLayer(args, Y, dicts, args.num_filter_maps)

    def forward(self, x, target, text_inputs):

        x = self.word_rep(x, target, text_inputs)

        x = x.transpose(1, 2)

        for conv in self.conv:
            x = conv(x)
        x = x.transpose(1, 2)

        y, loss = self.output_layer(x, target, text_inputs)

        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class MultiResCNN(nn.Module):

    def __init__(self, args, Y, dicts):
        super(MultiResCNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        self.conv = nn.ModuleList()
        filter_sizes = args.filter_size.split(',')

        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(self.word_rep.feature_size, self.word_rep.feature_size, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            xavier_uniform(tmp.weight)
            one_channel.add_module('baseconv', tmp)

            conv_dimension = self.word_rep.conv_dict[args.conv_layer]
            for idx in range(args.conv_layer):
                tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True,
                                    args.dropout)
                one_channel.add_module('se-resconv-{}'.format(idx), tmp)

            self.conv.add_module('channel-{}'.format(filter_size), one_channel)

        self.output_layer = OutputLayer(args, Y, dicts, self.filter_num * args.num_filter_maps)

    def forward(self, x, target, mask):

        # x = self.word_rep(x, target, text_inputs)
        x = self.word_rep(x, target)

        x = x.transpose(1, 2)

        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)

        y, loss = self.output_layer(x, target, mask)

        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class MultiResCNN_atten(nn.Module):

    def __init__(self, args, Y, dicts, cornet_dim=1000, n_cornet_blocks=2):
        super(MultiResCNN_atten, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        self.conv = nn.ModuleList()
        filter_sizes = args.filter_size.split(',')

        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(self.word_rep.feature_size, self.word_rep.feature_size, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            xavier_uniform(tmp.weight)
            one_channel.add_module('baseconv', tmp)

            conv_dimension = self.word_rep.conv_dict[args.conv_layer]
            for idx in range(args.conv_layer):
                tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True,
                                    args.dropout)
                one_channel.add_module('se-resconv-{}'.format(idx), tmp)

            self.conv.add_module('channel-{}'.format(filter_size), one_channel)

        # label graph
        # self.gcn = LabelNet(256, self.filter_num * args.num_filter_maps, args.embedding_size)

        # self.output_layer = OutputLayer(args, Y, dicts, self.filter_num * args.num_filter_maps)
        self.output_layer = OutputLayer(args, Y, dicts, self.filter_num * args.num_filter_maps)

        # corNet
        self.cornet = CorNet(Y, cornet_dim, n_cornet_blocks)

        # loss
        self.loss_function = nn.BCEWithLogitsLoss()

    def forward(self, x, target, mask, g, g_node_feature):
        # label_feature = self.gcn(g, g_node_feature)  # size: (bs, num_label, 100)
        # label_feature = torch.cat((label_feature, g_node_feature), dim=1)  # torch.Size([num_label, 200])

        # atten_mask = label_feature.transpose(0, 1) * mask.unsqueeze(1)
        # print('mask', atten_mask.size())

        # x = self.word_rep(x, target, text_inputs)
        x = self.word_rep(x, target)

        x = x.transpose(1, 2)

        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)
        # print('x', x.size())

        y = self.output_layer(x, target, mask)
        y = self.cornet(y)
        loss = self.loss_function(y, target)

        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class MultiResCNN_GCN(nn.Module):
    def __init__(self, args, Y, dicts, num_class, cornet_dim=1000, n_cornet_blocks=2):
        super(MultiResCNN_GCN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        self.conv = nn.ModuleList()
        filter_sizes = args.filter_size.split(',')

        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(self.word_rep.feature_size, self.word_rep.feature_size, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            xavier_uniform(tmp.weight)
            one_channel.add_module('baseconv', tmp)

            conv_dimension = self.word_rep.conv_dict[args.conv_layer]
            for idx in range(args.conv_layer):
                tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True, args.dropout)
                one_channel.add_module('resconv-{}'.format(idx), tmp)

            self.conv.add_module('channel-{}'.format(filter_size), one_channel)

        # self.U = nn.Linear(args.num_filter_maps * self.filter_num, args.embedding_size*2)
        # nn.init.xavier_uniform_(self.U.weight)
        self.U = nn.Linear(args.num_filter_maps * self.filter_num, Y)
        xavier_uniform(self.U.weight)

        # label graph
        self.gcn = LabelNet(256, args.embedding_size*2, args.embedding_size)

        # corNet
        self.cornet = CorNet(num_class, cornet_dim, n_cornet_blocks)

        self.output_layer = OutputLayer(args, num_class, dicts, self.filter_num * args.num_filter_maps)

        # loss
        self.loss_function = nn.BCEWithLogitsLoss()

    def forward(self, x, target, mask, g, g_node_feature):

        label_feature = self.gcn(g, g_node_feature)  # size: (bs, num_label, 100)

        label_feature = torch.cat((label_feature, g_node_feature), dim=1)  # torch.Size([num_label, 300])


        # atten_mask = label_feature.transpose(0, 1) * mask.unsqueeze(1)
        # atten_mask = g_node_feature.transpose(0, 1) * mask.unsqueeze(1)
        # print('mask', atten_mask.size())

        x = self.word_rep(x, target)
        x = x.transpose(1, 2)

        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            # atten = torch.softmax(torch.matmul(tmp, atten_mask), dim=1)
            # atten_tmp = torch.matmul(tmp.transpose(1, 2), atten).transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)  # size: (bs, num_label, 50 * len(ksz_list))

        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
        m = alpha.matmul(x) # [bs, Y, dim]
        m = m.transpose(1, 2) * mask.unsqueeze(1)
        m = m.transpose(1, 2)  # size: (bs, num_label, 50 * len(ksz_list))

        # x = self.U(x)
        # print('x', x.size())
        # atten = torch.softmax(torch.matmul(x, atten_mask), dim=1)
        # atten_x = torch.matmul(x.transpose(1, 2), atten).transpose(1, 2)

        # feature = torch.sum(x * label_feature, dim=2)
        y = torch.sum(m * label_feature, dim=2)
        y = self.cornet(y)
        #
        # y = self.output_layer(x, target)
        y = self.cornet(y)
        loss = self.loss_function(y, target)

        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class RNN_GCN(nn.Module):
    def __init__(self, args, Y, dicts, num_class, cornet_dim=1000, n_cornet_blocks=2):
        super(RNN_GCN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)
        self.rnn = nn.LSTM(input_size=args.embedding_size, hidden_size=args.embedding_size,
                           num_layers=args.rnn_num_layers, dropout=self.dropout if args.rnn_num_layers > 1 else 0,
                           bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(args.dropout)

        self.gcn = LabelNet(args.embedding_size, args.embedding_size, args.embedding_size)

        self.cornet = CorNet(num_class, cornet_dim, n_cornet_blocks)

        # loss
        self.loss_function = nn.BCEWithLogitsLoss()

    def forward(self, x, x_length, target, mask, g, g_node_feature):
        # Get label embeddings:
        label_feature = self.gcn(g, g_node_feature)
        label_feature = torch.cat((label_feature, g_node_feature), dim=1)  # torch.Size([num_label, 100*2])
        atten_mask = label_feature.transpose(0, 1) * mask.unsqueeze(1)

        x = self.word_rep(x, target)
        # x = x.transpose(1, 2)
        x = pack_padded_sequence(x, x_length, batch_first=True, enforce_sorted=False)  # packed input title
        x, (_,_) = self.rnn(x)
        x, _ = pad_packed_sequence(x, batch_first=True)

        x_atten = torch.softmax(torch.matmul(x, atten_mask), dim=1)
        x_feature = torch.matmul(x.transpose(1, 2), x_atten).transpose(1, 2)

        x_feature = torch.sum(x_feature * label_feature, dim=2)
        y = self.cornet(x_feature)
        loss = self.loss_function(y, target)
        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        xavier_uniform(self.conv1.weight)
        xavier_uniform(self.conv2.weight)
        if self.downsample is not None:
            xavier_uniform(self.downsample.weight)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DCAN(nn.Module):
    def __init__(self, args, Y, dicts):
        super(DCAN, self).__init__()
        self.configs = args
        self.word_rep = WordRep(args, Y, dicts)
        num_chans = [args.nhid] * args.levels
        self.tcn = TemporalConvNet(self.word_rep.feature_size, num_chans, args.kernel_size, args.dropout)
        self.lin = nn.Linear(num_chans[-1], args.nproj)
        self.output_layer = OutputLayer_label_smooth(args, Y, dicts, args.nproj)

        xavier_uniform(self.lin.weight)

    def forward(self, data, target, text_inputs=None):
        # data: [bs, len]
        bs, seq_len = data.size(0), data.size(1)
        x = self.word_rep(data, target)   # [bs, seq_len, dim_embed]
        hid_seq = self.tcn(x.transpose(1, 2)).transpose(1, 2)   # [bs, seq_len, nhid]
        hid_seq = F.relu(self.lin(hid_seq))
        logits, loss = self.output_layer(hid_seq, target, None)
        return logits, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class DilatedCNN(nn.Module):
    def __init__(self, args, Y, dicts, use_res=True):
        super(DilatedCNN, self).__init__()
        self.word_rep = WordRep(args, Y, dicts)

        self.dconv = nn.Sequential(nn.Conv1d(args.embedding_size, args.embedding_size, kernel_size=3, padding=(3-1) * 1, dilation=1),
                                   nn.SELU(), nn.AlphaDropout(p=0.05),
                                   nn.Conv1d(args.embedding_size, args.embedding_size, kernel_size=3, padding=(3-1) * 2, dilation=2),
                                   nn.SELU(), nn.AlphaDropout(p=0.05),
                                   nn.Conv1d(args.embedding_size, args.embedding_size, kernel_size=3, padding=(3-1) * 3, dilation=3),
                                   nn.SELU(), nn.AlphaDropout(p=0.05))

        self.use_res = use_res
        if self.use_res:
            self.shortcut = nn.Sequential(nn.Conv1d(args.embedding_size, args.embedding_size, kernel_size=1),
                                          nn.BatchNorm1d(args.embedding_size)
                                          )

        self.dropout = nn.Dropout(p=args.dropout)

        self.U = nn.Linear(args.embedding_size, Y)
        xavier_uniform(self.U.weight)

        self.final = nn.Linear(args.embedding_size, Y)
        xavier_uniform(self.final.weight)

        self.loss_function = nn.BCEWithLogitsLoss()

    def forward(self, x, target, mask):

        x = self.word_rep(x, target)
        x = x.transpose(1, 2)  # (bs, emb_dim, seq_length)
        print('embedding', x.size())
        out = self.dconv(x)  # (bs, embed_dim, seq_len-ksz+1)
        print('conv', out.size())

        if self.use_res:
            print('short', self.shortcut(x).size())
            out += self.shortcut(x)
        out = torch.tanh(out)
        out = self.dropout(out)
        out = out.transpose(1, 2)

        # alpha = torch.softmax(torch.matmul(x.transpose(1, 2), self.U.weight.transpose(0, 1)), dim=1)
        alpha = F.softmax(self.U.weight.matmul(out.transpose(1, 2)), dim=2)
        m = alpha.matmul(out)

        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        loss = self.loss_function(y, target)

        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class rnn_encoder(nn.Module):

    def __init__(self, args, Y, dicts):
        super(rnn_encoder, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)
        self.rnn = nn.LSTM(input_size=args.embedding_size, hidden_size=args.embedding_size,
                           num_layers=args.rnn_num_layers, dropout=args.dropout if args.rnn_num_layers > 1 else 0,
                           bidirectional=True, batch_first=True)

        self.dconv = nn.Sequential(nn.Conv1d(args.embedding_size, args.embedding_size, kernel_size=3, padding=0, dilation=1),
                                   nn.SELU(), nn.AlphaDropout(p=0.05),
                                   nn.Conv1d(args.embedding_size, args.embedding_size, kernel_size=3, padding=0, dilation=2),
                                   nn.SELU(), nn.AlphaDropout(p=0.05),
                                   nn.Conv1d(args.embedding_size, args.embedding_size, kernel_size=3, padding=0, dilation=3),
                                   nn.SELU(), nn.AlphaDropout(p=0.05))

    def forward(self, x, x_length, target):

        x = self.word_rep(x, target)
        rnn = pack_padded_sequence(x, x_length, batch_first=True, enforce_sorted=False)  # packed input title
        rnn, state = self.rnn(rnn)
        rnn, _ = pad_packed_sequence(rnn, batch_first=True)
        rnn = rnn[:, :, :100] + rnn[:, :, 100:]  # size: (bs, seq_len, emb_dim)

        conv = rnn.permute(0, 2, 1)
        conv = self.dconv(conv)  # (bs, embed_dim, seq_len-ksz+1)
        conv = conv.transpose(1, 2)

        state = (state[0][::2], state[1][::2])

        return rnn, state, conv


class RNN_DCNN(nn.Module):
    def __init__(self, args, Y, dicts, cornet_dim=1000, n_cornet_blocks=2):
        super(RNN_DCNN, self).__init__()

        self.encoder = rnn_encoder(args, Y, dicts)

        # self.U = nn.Linear(args.embedding_size, Y)
        # xavier_uniform(self.U.weight)

        # self.gcn = LabelNet(args.embedding_size, args.embedding_size, args.embedding_size)
        self.cornet = CorNet(Y, cornet_dim, n_cornet_blocks)

        self.final = nn.Linear(args.embedding_size*2, Y)
        xavier_uniform(self.final.weight)

        self.loss_function = nn.BCEWithLogitsLoss()

    def forward(self, x, x_length, target, mask, g, g_node_feature):
        rnn, state, conv = self.encoder(x, x_length, target)

        # label_feature = self.gcn(g, g_node_feature)
        # label_feature = torch.cat((label_feature, g_node_feature), dim=1)  # torch.Size([num_label, 100*2])
        atten_mask = g_node_feature.transpose(0, 1) * mask.unsqueeze(1)

        # alpha_rnn = F.softmax(self.U.weight.matmul(rnn.transpose(1, 2)), dim=2)
        # m_rnn = alpha_rnn.matmul(rnn)
        #
        # alpha_conv = F.softmax(self.U.weight.matmul(conv.transpose(1, 2)), dim=2)
        # m_conv = alpha_conv.matmul(conv)
        alpha_rnn = torch.softmax(torch.matmul(rnn, atten_mask), dim=1)
        m_rnn = torch.matmul(rnn.transpose(1, 2), alpha_rnn).transpose(1, 2)

        alpha_conv = torch.softmax(torch.matmul(conv, atten_mask), dim=1)
        m_conv = torch.matmul(conv.transpose(1, 2), alpha_conv).transpose(1, 2)

        m = torch.cat((m_rnn, m_conv), dim=2)

        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        loss = self.loss_function(y, target)
        return y, loss


def pick_model(args, dicts, num_class):
    # Y = len(dicts['ind2c'])
    if args.model == 'CNN':
        model = CNN(args, num_class, dicts)
    elif args.model == 'MultiCNN':
        model = MultiCNN(args, num_class, dicts)
    elif args.model == 'ResCNN':
        model = ResCNN(args, num_class, dicts)
    elif args.model == 'MultiResCNN':
        model = MultiResCNN(args, num_class, dicts)
    elif args.model == 'MultiResCNN_GCN':
        model = MultiResCNN_GCN(args, num_class, dicts, num_class)
    elif args.model == 'MultiSeResCNN_GCN':
        model = MultiResCNN_GCN(args, num_class, dicts, num_class)
    elif args.model == 'RNN_GCN':
        model = RNN_GCN(args, num_class, dicts, num_class)
    elif args.model == 'DCAN':
        model = DCAN(args, num_class, dicts)
    elif args.model == 'MultiResCNN_atten':
        model = MultiResCNN_atten(args, num_class, dicts)
    elif args.model == 'DilatedCNN':
        model = DilatedCNN(args, num_class, dicts)
    elif args.model == 'RNN_DCNN':
        model = RNN_DCNN(args, num_class, dicts)
    else:
        raise RuntimeError("wrong model name")

    if args.test_model:
        sd = torch.load(args.test_model)
        model.load_state_dict(sd)
    if args.gpu >= 0:
        model.cuda(args.gpu)
    return model

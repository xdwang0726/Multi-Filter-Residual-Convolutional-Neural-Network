import argparse
import csv
import io
import os
import sys
import timeit

import dgl
import numpy as np
import pandas as pd
import torch
from dgl.data.utils import save_graphs
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vectors
from tqdm import tqdm

tokenizer = get_tokenizer('basic_english')


def unicode_csv_reader(unicode_csv_data, **kwargs):
    r"""Since the standard csv library does not handle unicode in Python 2, we need a wrapper.
    Borrowed and slightly modified from the Python docs:
    https://docs.python.org/2/library/csv.html#csv-examples
    Args:
        unicode_csv_data: unicode csv data (see example below)
    Examples:
        >>> from torchtext.utils import unicode_csv_reader
        >>> import io
        >>> with io.open(data_path, encoding="utf8") as f:
        >>>     reader = unicode_csv_reader(f)
    """

    # Fix field larger than field limit error
    maxInt = sys.maxsize
    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)
    csv.field_size_limit(maxInt)

    for line in csv.reader(unicode_csv_data, **kwargs):
        yield line


def get_ICD_codes(MIMIC_3_DIR):
    df = pd.read_csv('%s/ALL_CODES.csv' % MIMIC_3_DIR, dtype={"ICD9_CODE": str})
    unique_codes = list(df['ICD9_CODE'].unique())

    icd_codes_and_name_pair = []
    with open('%sdata/ICD9_descriptions.txt' % MIMIC_3_DIR, 'r') as f:
        for line in f:
            (key, value) = line.split('\t')
            if key in unique_codes:
                icd_codes_and_name_pair.append((key, value.strip()))

    f = open('%sdata/icd/MIMIC_icd_codes_name_pair.txt' % MIMIC_3_DIR, 'w')
    for t in icd_codes_and_name_pair:
        line = '\t'.join(str(x) for x in t)
        f.write(line + '\n')
    f.close()


def cooccurence_node_edge(train_data_path, id_pair_file, threshold, vectors):
    # get icd in each example
    icd_label = []
    with io.open(train_data_path, encoding="utf8") as f:
        next(f)
        reader = unicode_csv_reader(f)
        for row in reader:
            icd_label.append(str(row[3]))

    # get descriptor and MeSH mapped
    mapping_id = {}
    with open(id_pair_file, 'r') as f:
        for line in f:
            (key, value) = line.split('\t')
            mapping_id[key] = value.strip()

    # count number of nodes and get parent and children edges
    print('count number of nodes and get edges of the graph')
    node_count = len(mapping_id)
    values = list(mapping_id.keys())

    # count the co-occurrence between MeSH
    cooccurrence_counts = {}
    for doc in icd_label:
        icds = doc.split(';')
        for label in icds:
            labelDict = dict()
            if label in cooccurrence_counts:
                labelDict = cooccurrence_counts[label]
            for target in icds:
                # if target != label:
                if target in labelDict:
                    labelDict[target] = labelDict[target] + 1
                else:
                    labelDict[target] = 1

            cooccurrence_counts[label] = labelDict

    cooccurrence_counts = {k: v for k, v in cooccurrence_counts.items() if k}
    cooccurrence_matrix = pd.DataFrame(cooccurrence_counts)
    label_union = cooccurrence_matrix.index.union(cooccurrence_matrix.columns)
    cooccurrence_matrix = cooccurrence_matrix.reindex(index=label_union, columns=label_union)
    cooccurrence_matrix = cooccurrence_matrix.fillna(0)  # replace all nan value to 0

    # calculate the frequency each label in the training set
    num = np.diag(cooccurrence_matrix).tolist()

    # get co-occurrence edges
    edge_frame = cooccurrence_matrix.div(num, axis='index')
    edge_frame = (edge_frame >= threshold) * 1  # replacing each element larger than threshold by 1 else 0

    edge_index = np.argwhere(edge_frame.values == 1)
    train_label_list = list(cooccurrence_matrix)
    edge_cooccurrence = []
    for i in edge_index:
        if train_label_list[i[0]] != train_label_list[i[1]]:
            item = (train_label_list[i[0]], train_label_list[i[1]])
            idex_item = (values.index(item[0]), values.index(item[1]))
            edge_cooccurrence.append(idex_item)

    print('get label embeddings')
    label_embedding = torch.zeros(0)
    for key, value in tqdm(mapping_id.items()):
        key = tokenizer(key)
        key = [k.lower() for k in key]
        key_embedding = torch.zeros(0)
        for k in key:
            embedding = vectors.__getitem__(k).reshape(1, 100)
            key_embedding = torch.cat((key_embedding, embedding), dim=0)
        key_embedding = torch.mean(input=key_embedding, dim=0, keepdim=True)
        label_embedding = torch.cat((label_embedding, key_embedding), dim=0)

    return edge_cooccurrence, node_count, label_embedding


def build_icd_graph(edge_list, nodes, label_embedding):
    print('start building the graph')
    g = dgl.DGLGraph()
    # add nodes into the graph
    print('add nodes into the graph')
    g.add_nodes(nodes)
    # add edges, directional graph
    print('add edges into the graph')
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    # add node features into the graph
    print('add node features into the graph')
    g.ndata['feat'] = label_embedding
    return g


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train')
    parser.add_argument('--threshold', type=float, default=1)
    parser.add_argument('--word2vec_path')
    parser.add_argument('--pair_path')
    parser.add_argument('--parent_children_path')
    parser.add_argument('--output')

    args = parser.parse_args()

    print('Load pre-trained vectors')
    cache, name = os.path.split(args.word2vec_path)
    vectors = Vectors(name=name, cache=cache)
    edges, node_count, label_embedding = cooccurence_node_edge(args.train, args.pair_path, args.threshold, vectors)
    G = build_icd_graph(edges, node_count, label_embedding)

    save_graphs(args.output, G)


if __name__ == "__main__":
    start = timeit.default_timer()
    main()
    stop = timeit.default_timer()
    print('Time: ', stop - start)
# fine-tune
# run pretrain.py first

import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import models
import utils
from tqdm import tqdm
from GraphSMOTE import GraphSMOTE

def fine_tune():
    encoder.train()
    classifier.train()
    decoder.train()
    optimizer_en.zero_grad()
    optimizer_cls.zero_grad()
    optimizer_de.zero_grad()

    # extract embeddings from features
    embed = encoder(features, adj)

    # number of original labels
    ori_num = labels.shape[0]
    
    # generate synthetic nodes
    embed, labels_new, idx_train_new, adj_up = GraphSMOTE(embed, labels, idx_train, adj.to_dense(), 1)
    
    # restore graph from embeddings
    generated_G = decoder(embed)

    # calculate loss between orginal and new adjacency matrix
    loss_edge = utils.adj_mse_loss(generated_G[:ori_num, :][:, :ori_num], adj.detach().to_dense())
    # adj = utils.dense_to_sparse(adj)

    threshold = 0.5
    adj_new = torch.where(generated_G < threshold, 0.0, 1.0)

    adj_new = torch.mul(adj_up, adj_new)
    adj_new[:ori_num, :][:, :ori_num] = adj.detach().to_dense()

    output = classifier(embed, adj_new)
    
    loss_node = F.cross_entropy(output[idx_train_new], labels_new[idx_train_new])
    
    loss = loss_node + loss_edge * args.weight

    loss.backward()
    
    optimizer_en.step()
    optimizer_cls.step()
    optimizer_de.step()

    f1 = utils.print_metrics(output[idx_train], labels[idx_train], print_flag=False)

    return f1

from refactor import refactor
def test():
    encoder.eval()
    classifier.eval()
    decoder.eval()

    embed = encoder(features, adj)
    output = classifier(embed, adj)
    
    utils.print_metrics(output[idx_test], labels[idx_test])

    mm_adj = decoder(embed)
    mm_adj = utils.elementwise_sparse_dense_multiply(mm_adj, adj)

    mm_adj = mm_adj.to_dense().cpu().detach().numpy()
    preds = output.max(1)[1].type_as(labels).cpu().detach().numpy()

    refactor(args.fine_tuned_project, preds, mm_adj, idx_test)


# keep model in memory
def keep_model():
    model = {}
    model['encoder'] = encoder.state_dict()
    model['decoder'] = decoder.state_dict()
    model['classifier'] = classifier.state_dict()
    return model

def load_model(file, encoder, decoder, classifier):
    loaded_content = torch.load(file, map_location=lambda storage, loc:storage)

    encoder.load_state_dict(loaded_content['encoder'])
    decoder.load_state_dict(loaded_content['decoder'])
    classifier.load_state_dict(loaded_content['classifier'])

    print("successfully loaded: " + file)
    return encoder, decoder, classifier


t_start = time.time()

# Training setting
parser = utils.get_parser()
args = parser.parse_args()

# load data
adj, features, labels = utils.load_data(args.fine_tuned_project)
idx_train, idx_val, idx_test = utils.split_labels(labels, train_ratio=args.fine_tune_data, val_ratio=0, test_ratio=1-args.fine_tune_data)
features, adj, labels = features.to(args.device), adj.to(args.device), labels.to(args.device)
feature_num = features.shape[1]
    
# Model and optimizer
encoder = models.SageEncoder(feature_num, args.hidden_dim, args.dropout)
classifier = models.SageClassifier(args.hidden_dim, args.hidden_dim, args.class_num, args.dropout)
decoder = models.GraphDecoder(args.hidden_dim)

# load pretrained model 
encoder, decoder, classifier = load_model(f'pretrained_models/{args.pretrained_project}.pth', encoder, decoder, classifier)

encoder = encoder.to(args.device)
classifier = classifier.to(args.device)
decoder = decoder.to(args.device)

optimizer_en = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimizer_cls = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimizer_de = optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)


if args.fine_tune_data == 0:
    test()
else:
    pbar = tqdm(range(args.fine_tune_epochs), total=args.fine_tune_epochs, ncols=100, unit="epoch", colour="red")
    best_f1 = 0
    for epoch in pbar:
        f1 = fine_tune()
        if f1 > best_f1:
            model_best = keep_model()
            best_f1 = f1
        
        pbar.set_postfix({"best f1": best_f1})

    encoder.load_state_dict(model_best['encoder'])
    decoder.load_state_dict(model_best['decoder'])
    classifier.load_state_dict(model_best['classifier'])
    test()


print("Total time elapsed: {:.4f}s".format(time.time() - t_start))



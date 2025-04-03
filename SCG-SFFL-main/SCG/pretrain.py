import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import models
import utils
from tqdm import tqdm
from GraphSMOTE import GraphSMOTE

def train():
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

# save model
def save_model(model, project):
    torch.save(model, 'pretrained_models/{}.pth'.format(project))


# keep model in memory
def keep_model():
    model = {}
    model['encoder'] = encoder.state_dict()
    model['decoder'] = decoder.state_dict()
    model['classifier'] = classifier.state_dict()
    return model

# Training setting
parser = utils.get_parser()
args = parser.parse_args()

projects = ['activemq', 'alluxio', 'binnavi', 'kafka', 'realm-java']
args.epochs = 1500
t_start = time.time()

for project in projects:
    # load data
    adj, features, labels = utils.load_data(project)
    idx_train, idx_val, idx_test = utils.split_labels(labels, train_ratio=1, val_ratio=0, test_ratio=0)
    features, adj, labels = features.to(args.device), adj.to(args.device), labels.to(args.device)
    feature_num = features.shape[1]
    
    # Model and optimizer
    encoder = models.SageEncoder(feature_num, args.hidden_dim, args.dropout)
    classifier = models.SageClassifier(args.hidden_dim, args.hidden_dim, args.class_num, args.dropout)
    decoder = models.GraphDecoder(args.hidden_dim)

    encoder = encoder.to(args.device)
    classifier = classifier.to(args.device)
    decoder = decoder.to(args.device)

    optimizer_en = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_cls = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_de = optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    pbar = tqdm(range(args.epochs), total=args.epochs, ncols=100, unit="epoch", colour="red")

    best_f1 = 0

    for epoch in pbar:
        f1 = train()
        if f1  > best_f1:
            model_best = keep_model()
            best_f1 = f1
        
        pbar.set_postfix({"best f1": best_f1})
   
    save_model(model_best, project)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_start))
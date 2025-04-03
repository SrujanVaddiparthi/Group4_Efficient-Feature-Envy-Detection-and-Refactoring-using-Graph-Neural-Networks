import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import models
import utils
from tqdm import tqdm

def train():
    encoder.train()
    classifier.train()
    decoder.train()
    optimizer_en.zero_grad()
    optimizer_cls.zero_grad()
    optimizer_de.zero_grad()

    # extract embeddings from features
    embed = encoder(features, adj)

    output = classifier(embed, adj)
    
    loss_node = F.cross_entropy(output[idx_train], labels[idx_train])
    
    loss = loss_node 

    loss.backward()
    
    optimizer_en.step()
    optimizer_cls.step()
    optimizer_de.step()

    f1 = utils.print_metrics(output[idx_val], labels[idx_val], print_flag=False)

    return f1


def test():
    encoder.eval()
    classifier.eval()
    decoder.eval()

    embed = encoder(features, adj)
    output = classifier(embed, adj)
    
    utils.print_metrics(output[idx_test], labels[idx_test])

    
# save model
def save_model(model, project):
    torch.save(model, 'checkpoint/{}.pth'.format(project))


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

# load data
adj, features, labels = utils.load_data(args.project)
feature_num = features.shape[1]

# get train, validation, test data split
idx_train, idx_val, idx_test = utils.split_labels(labels, random_seed=args.random_seed)

features = features.to(args.device)
adj = adj.to(args.device)
labels = labels.to(args.device)

t_start = time.time()
best_f1 = 0
pbar = tqdm(range(args.epochs), total=args.epochs, leave=False, 
            ncols=100, unit="epoch", unit_scale=False, colour="red")

for _ in range(args.repeat_time):
    # Model and optimizer
    encoder = models.SageEncoder(feature_num, args.hidden_dim, args.dropout)

    classifier = models.SageClassifier(args.hidden_dim, args.hidden_dim, args.class_num, args.dropout)

    decoder = models.GraphDecoder(args.hidden_dim)

    optimizer_en = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_cls = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_de = optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    encoder = encoder.to(args.device)
    classifier = classifier.to(args.device)
    decoder = decoder.to(args.device)

    pbar = tqdm(range(args.epochs), total=args.epochs, ncols=100, unit="epoch", colour="red")
    this_time_best_f1 = 0
    for epoch in pbar:
        f1 = train()
        if f1  > best_f1:
            model_best = keep_model()
            best_f1 = f1
        if f1 > this_time_best_f1:
            this_time_best_f1 = f1
        
        pbar.set_postfix({"best f1 in val this time": this_time_best_f1, 'best f1 in val': best_f1})
   

save_model(model_best, args.project)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_start))

# Testing
encoder.load_state_dict(model_best['encoder'])
decoder.load_state_dict(model_best['decoder'])
classifier.load_state_dict(model_best['classifier'])
test()
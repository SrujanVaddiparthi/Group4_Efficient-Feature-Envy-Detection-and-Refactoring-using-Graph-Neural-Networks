import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from models import GNNReconstructor, PositionEncoder, SemanticEncoder
import utils
from tqdm import tqdm
import pickle

def test():
    model.eval()

    with torch.no_grad():
        output = model(mc_own_adj, mc_call_adj, method_embeddings, class_embeddings)

        class_pred = output.argmax(dim=1)
        class_true = fmc_own_adj.indices()[1, :].flatten()
        class_real = mc_own_adj.indices()[1, :].flatten()

        print('Train set')
        utils.print_metrics(class_pred[train_idx], class_true[train_idx], class_real[train_idx], True)

        print('Test set')
        utils.print_metrics(class_pred[test_idx], class_true[test_idx], class_real[test_idx], True)

def fine_tune():
    model.train()
    optimizer.zero_grad()

    # calculate loss between orginal and new adjacency matrix
    output = model(mc_own_adj, mc_call_adj, method_embeddings, class_embeddings)
    
    loss = F.cross_entropy(output[train_idx,:], fmc_own_adj.indices()[1,train_idx])
    
    loss.backward() # Compute gradients
    optimizer.step() # Update weights

    # validate
    class_pred = output.argmax(dim=1)
    class_true = fmc_own_adj.indices()[1, :].flatten()
    class_real = mc_own_adj.indices()[1, :].flatten()
    acc, precision, recall, f1 = utils.print_metrics(class_pred[train_idx], class_true[train_idx], class_real[train_idx])
    
    return f1.item()


# Training setting
t_start = time.time()

parser = utils.get_parser()
args = parser.parse_args()
args.hidden_dim = 256

print("=== Tunable Parameters ===")
for arg in vars(args):
    print(arg, ":", getattr(args, arg))
print()
torch.manual_seed(args.random_seed)

device = args.device
# load data
mc_own_adj, mc_call_adj, fmc_own_adj, class_tokens, method_tokens, labels = utils.load_data(args.fine_tuned_project)

class_num, method_num = len(class_tokens), len(method_tokens)

# get train, validation, test data split
train_idx, val_idx, test_idx = utils.split_dataset(labels, train_ratio=args.fine_tune_data, val_ratio=0, test_ratio=1-args.fine_tune_data)

# encoding 
print('Both position encoding and semantic encoding are taken.\n')
method_position_encoder = PositionEncoder(method_num, args.hidden_dim)
class_position_encoder = PositionEncoder(class_num, args.hidden_dim)
semantic_encoder = SemanticEncoder(method_tokens+class_tokens, args.hidden_dim, args.word_embedding_epochs, args.random_seed)
t = time.time()
method_position_embeddings = method_position_encoder().to(device)
class_position_embeddings = class_position_encoder().to(device)
print('Position encoding is completed, taking {:.2f} seconds.'.format(time.time() - t))
t = time.time()
method_semantic_embeddings = semantic_encoder(method_tokens).to(device)
class_semantic_embeddings = semantic_encoder(class_tokens).to(device)
print('Semantic encoding is completed, taking {:.2f} seconds.\n'.format(time.time() - t))
method_embeddings = method_position_embeddings + method_semantic_embeddings
class_embeddings = class_position_embeddings + class_semantic_embeddings

mc_own_adj, mc_call_adj, fmc_own_adj = mc_own_adj.to(device), mc_call_adj.to(device), fmc_own_adj.to(device)

# Model and optimizer
model = GNNReconstructor(args.hidden_dim, args.conv, args.head_num, args.aggr, args.dropout)
model.load_state_dict(torch.load(f'pretrained_models/{args.pretrained_project}.pt'))
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.fine_tune_data == 0:
    test()
else:
    pbar = tqdm(range(args.fine_tune_epochs), total=args.fine_tune_epochs, ncols=100, unit="epoch", colour="red")
    best_f1 = 0
    for epoch in pbar:
        f1 = fine_tune()
        if f1 > best_f1:
            torch.save(model.state_dict(), 'best_model.pt')
            best_f1 = f1
        
        pbar.set_postfix({"best f1": best_f1})

    model.load_state_dict(torch.load('best_model.pt'))

    test()

print("Total time elapsed: {:.4f}s".format(time.time() - t_start))

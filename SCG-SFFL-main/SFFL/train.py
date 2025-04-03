import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from models import GNNReconstructor, PositionEncoder, SemanticEncoder
import utils
from tqdm import tqdm


def test():
    model.eval()

    with torch.no_grad():
        output = model(mc_own_adj, mc_call_adj, method_embeddings, class_embeddings)

        class_pred = output.argmax(dim=1)
        class_true = fmc_own_adj.indices()[1, :].flatten()
        class_real = mc_own_adj.indices()[1, :].flatten()

        print('Train set')
        utils.print_metrics(class_pred[train_idx], class_true[train_idx], class_real[train_idx], True)

        print('Validation set')
        utils.print_metrics(class_pred[val_idx], class_true[val_idx], class_real[val_idx], True)

        print('Test set')
        utils.print_metrics(class_pred[test_idx], class_true[test_idx], class_real[test_idx], True)

def train():
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
    acc, precision, recall, f1 = utils.print_metrics(class_pred[val_idx], class_true[val_idx], class_real[val_idx])
    
    return f1.item()


# Training setting
t_start = time.time()

parser = utils.get_parser()
args = parser.parse_args()
if args.project == "activemq":
    args.weight_decay = 8e-5
print("=== Tunable Parameters ===")
for arg in vars(args):
    print(arg, ":", getattr(args, arg))
print()
torch.manual_seed(args.random_seed)

device = args.device
# load data
mc_own_adj, mc_call_adj, fmc_own_adj, class_tokens, method_tokens, labels = utils.load_data(args.project)

class_num, method_num = len(class_tokens), len(method_tokens)

# get train, validation, test data split
train_idx, val_idx, test_idx = utils.split_dataset(labels, 0.6, 0.2, 0.2, args.random_seed)

# encoding 
if args.encoding == 1:
    print('Both position encoding and semantic encoding are taken.\n')
    method_position_encoder = PositionEncoder(method_num, args.hidden_dim)
    class_position_encoder = PositionEncoder(class_num, args.hidden_dim)
    semantic_encoder = SemanticEncoder(class_tokens+method_tokens, args.hidden_dim, args.word_embedding_epochs, args.random_seed)
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

elif args.encoding == 2:
    print('Only position encoding encoding is taken.\n')
    method_position_encoder = PositionEncoder(method_num, args.hidden_dim)
    class_position_encoder = PositionEncoder(class_num, args.hidden_dim)
    t = time.time()
    method_embeddings = method_position_encoder().to(device)
    class_embeddings = class_position_encoder().to(device)
    print('Position encoding is completed, taking {:.2f} seconds.\n'.format(time.time() - t))

else:
    print('Only semantic encoding encoding is taken.')
    semantic_encoder = SemanticEncoder(class_tokens+method_tokens, args.hidden_dim, args.word_embedding_epochs, args.random_seed)
    t = time.time()
    method_embeddings = semantic_encoder(method_tokens).to(device)
    class_embeddings = semantic_encoder(class_tokens).to(device)
    print('Semantic encoding is completed, taking {:.2f} seconds.\n'.format(time.time() - t))


mc_own_adj, mc_call_adj, fmc_own_adj = mc_own_adj.to(device), mc_call_adj.to(device), fmc_own_adj.to(device)

t_train = time.time()
print('Start training...\n')
best_f1, this_time_best_f1 = 0, 0

for _ in range(args.repeat_time):
    # Model and optimizer
    model = GNNReconstructor(args.hidden_dim, args.conv, args.head_num, args.aggr, args.dropout)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    pbar = tqdm(range(args.epochs), total=args.epochs, ncols=100, unit="epoch", colour="red")
    for epoch in pbar:
        f1 = train()
        if f1  > best_f1:
            torch.save(model.state_dict(), 'best_model.pt')
            best_f1 = f1
        if f1 > this_time_best_f1:
            this_time_best_f1 = f1
        
        pbar.set_postfix({"best f1 in val this time": this_time_best_f1, 'best f1 in val': best_f1})
    this_time_best_f1 = 0

print('Training finished.')
print('Best f1: {:.2%}'.format(best_f1))
print('Train time: {:.2f}'.format(time.time() - t_train))
print('Total time: {:.2f}\n'.format(time.time() - t_start))
model.load_state_dict(torch.load('best_model.pt'))
test()
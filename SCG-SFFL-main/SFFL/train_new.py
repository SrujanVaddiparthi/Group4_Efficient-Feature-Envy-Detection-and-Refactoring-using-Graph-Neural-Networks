import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from models import GNNReconstructor, PositionEncoder, SemanticEncoder
import utils_new as utils
import csv
import os
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
    output = model(mc_own_adj, mc_call_adj, method_embeddings, class_embeddings)
    loss = F.cross_entropy(output[train_idx, :], fmc_own_adj.indices()[1, train_idx])
    loss.backward()
    optimizer.step()
    class_pred = output.argmax(dim=1)
    class_true = fmc_own_adj.indices()[1, :].flatten()
    class_real = mc_own_adj.indices()[1, :].flatten()
    acc, precision, recall, f1 = utils.print_metrics(class_pred[val_idx], class_true[val_idx], class_real[val_idx])
    return f1.item()

def save_metrics(project, hidden_dim, word_embedding_epochs, acc, f1, auc):
    output_file = "rq1_results.csv"
    header = ["Project", "VectorSize", "Epochs", "Accuracy", "F1Score", "AUC"]
    write_header = not os.path.exists(output_file)

    with open(output_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([project, hidden_dim, word_embedding_epochs, acc, f1, auc])

def run_experiment(args):
    global model, optimizer, mc_own_adj, mc_call_adj, fmc_own_adj
    global method_embeddings, class_embeddings
    global train_idx, val_idx, test_idx

    t_start = time.time()
    if args.project == "activemq":
        args.weight_decay = 8e-5
    print("=== Tunable Parameters ===")
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    print()
    torch.manual_seed(args.random_seed)
    device = args.device
    mc_own_adj, mc_call_adj, fmc_own_adj, class_tokens, method_tokens, labels = utils.load_data(args.project)
    class_num, method_num = len(class_tokens), len(method_tokens)
    train_idx, val_idx, test_idx = utils.split_dataset(labels, 0.6, 0.2, 0.2, args.random_seed)

    if args.encoding == 1:
        method_position_encoder = PositionEncoder(method_num, args.hidden_dim)
        class_position_encoder = PositionEncoder(class_num, args.hidden_dim)
        semantic_encoder = SemanticEncoder(class_tokens+method_tokens, args.hidden_dim, args.word_embedding_epochs, args.random_seed)
        method_position_embeddings = method_position_encoder().to(device)
        class_position_embeddings = class_position_encoder().to(device)
        method_semantic_embeddings = semantic_encoder(method_tokens).to(device)
        class_semantic_embeddings = semantic_encoder(class_tokens).to(device)
        method_embeddings = method_position_embeddings + method_semantic_embeddings
        class_embeddings = class_position_embeddings + class_semantic_embeddings
    elif args.encoding == 2:
        method_position_encoder = PositionEncoder(method_num, args.hidden_dim)
        class_position_encoder = PositionEncoder(class_num, args.hidden_dim)
        method_embeddings = method_position_encoder().to(device)
        class_embeddings = class_position_encoder().to(device)
    else:
        semantic_encoder = SemanticEncoder(class_tokens+method_tokens, args.hidden_dim, args.word_embedding_epochs, args.random_seed)
        method_embeddings = semantic_encoder(method_tokens).to(device)
        class_embeddings = semantic_encoder(class_tokens).to(device)

    mc_own_adj, mc_call_adj, fmc_own_adj = mc_own_adj.to(device), mc_call_adj.to(device), fmc_own_adj.to(device)

    best_f1 = 0
    patience = 10  # Early stopping patience
    wait = 0

    for _ in range(args.repeat_time):
        model = GNNReconstructor(args.hidden_dim, args.conv, args.head_num, args.aggr, args.dropout)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        pbar = tqdm(range(args.epochs), total=args.epochs, ncols=100, unit="epoch", colour="red")

        for epoch in pbar:
            f1 = train()

            if f1 > best_f1:
                torch.save(model.state_dict(), 'best_model.pt')
                best_f1 = f1
                wait = 0
            else:
                wait += 1

            pbar.set_postfix({"F1(val)": f"{f1:.4f}", "Best F1": f"{best_f1:.4f}", "Wait": wait})

            if wait >= patience:
                print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    with torch.no_grad():
        output = model(mc_own_adj, mc_call_adj, method_embeddings, class_embeddings)
        class_pred = output.argmax(dim=1)
        class_true = fmc_own_adj.indices()[1, :].flatten()
        class_real = mc_own_adj.indices()[1, :].flatten()
        acc, precision, recall, f1 = utils.print_metrics(class_pred[test_idx], class_true[test_idx], class_real[test_idx])

    return acc.item(), f1.item(), None

if __name__ == "__main__":
    parser = utils.get_parser()
    args = parser.parse_args()
    run_experiment(args)

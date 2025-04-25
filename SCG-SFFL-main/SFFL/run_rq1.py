import argparse
from train_new import run_experiment, save_metrics
import os
import utils_new as utils


def main():
    parser = utils.get_parser()
    args = parser.parse_args()

    projects = ["activemq", "alluxio", "binnavi", "kafka", "realm-java"]
    vector_sizes = [64, 128, 256]
    epochs_list = [50, 100, 150]

    for project in projects:
        gt_path = os.path.join("SCG-SFFL-main", "SFFL", "data", project, "ground_truth.csv")
        if not os.path.exists(gt_path):
            print(f"⚠️ Skipping {project}: ground_truth.csv not found at {gt_path}.")
            continue

        for vec_size in vector_sizes:
            for epoch in epochs_list:
                print(f"\n🔍 Running for project={project}, vector_size={vec_size}, epochs={epoch}")
                args.project = project
                args.hidden_dim = vec_size
                args.word_embedding_epochs = epoch
                args.epochs = epoch
                try:
                    acc, f1, auc = run_experiment(args)
                    save_metrics(project, vec_size, epoch, acc, f1, auc)
                except Exception as e:
                    print(f"❌ Error running {project} with vec_size={vec_size}, epoch={epoch}: {e}")

if __name__ == "__main__":
    main()

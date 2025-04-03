import os
import time
import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
import statistics
project_list = ['activemq', 'alluxio', 'binnavi', 'kafka', 'realm-java']

hidden_dim_list = [128, 256, 512]
epoch_list = [200, 300, 400]
random_seed_list = list(range(5))
repeat_time = 5
t_start = time.time()

for project in project_list:
    for hidden_dim in hidden_dim_list:
        for epoch in epoch_list:
            for random_seed in random_seed_list:
                filename = f"RQ1/{project}_{hidden_dim}_{epoch}_{random_seed}.txt"
                    
                if os.path.exists(filename):
                    print(f"{filename} already exists, skipping...")
                    continue
                t_round = time.time()
                command = f"python train.py --project {project} --hidden_dim {hidden_dim} --word_embedding_epochs {epoch} --random_seed {random_seed} --repeat_time {repeat_time}"
                    
                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                stdout, _ = process.communicate()

                with open(filename, "w") as file:
                    file.write(stdout.decode())
                    print(f'Results have been saved to {filename}')
                print('This round time:', time.time() - t_round)

print('Total time:', time.time() - t_start)
f1_scores = {}

for project in project_list:
    for hidden_dim in hidden_dim_list:
        for epoch in epoch_list:
            f1_score_list = []
            for random_seed in random_seed_list:
                filename = f"RQ1/{project}_{hidden_dim}_{epoch}_{random_seed}.txt"

                if os.path.exists(filename):
                    with open(filename, "r") as file:
                        content = file.read()

                    start_index = content.find("Test set\n") + len("Test set\n")
                    results = content[start_index:]

                    f1_score_match = re.search(r"F1-Score2:\s+(\d+\.\d+)%", results)
                    
                    if f1_score_match is not None:
                        f1_score = float(f1_score_match.group(1))
                        f1_score_list.append(f1_score)

            f1_score_list = sorted(f1_score_list)
            f1_scores[(project, hidden_dim, epoch)] = statistics.mean(f1_score_list) #f1_score_list[-1]#
            print(f"Test set results for {project}_{hidden_dim}_{epoch}:")
            print(f"F1 Score: {statistics.mean(f1_score_list)}")
            print()

fig, axs = plt.subplots(1, len(hidden_dim_list), figsize=(30, 10))

min_f1_score = float("inf")
max_f1_score = float("-inf")
line_styles = ['-', '--', '-.', ':', '-']
marker_styles = ['o', 's', '^', 'v', 'x']
project_dic = {'binnavi': 'BinNavi', 'activemq': 'ActiveMQ', 'kafka': 'Kafka', 'alluxio': 'Alluxio', 'realm-java': 'Realm-java'}
for i, hidden_dim in enumerate(hidden_dim_list):
    for j, project in enumerate(project_list):
        # 提取每个项目对应的 F1 结果
        f1_scores_project = [f1_scores[(project, hidden_dim, epoch)] for epoch in epoch_list]

        # 绘制折线图
        axs[i].plot(epoch_list, f1_scores_project, linestyle=line_styles[j], marker=marker_styles[j], label=f"{project_dic[project]}")
        min_f1_score = min(min_f1_score, min(f1_scores_project))
        max_f1_score = max(max_f1_score, max(f1_scores_project))

    axs[i].set_title(f"Vector Size: {hidden_dim}")
    axs[i].set_xlabel("Epoch")
    
plt.subplots_adjust(wspace=0)

# 设置统一的纵坐标轴
for ax in axs:
    ax.set_ylim(min_f1_score-2, 100)
plt.setp(axs[1:], yticks=[])
plt.rcParams["text.usetex"] = True
axs[0].set_ylabel(r'$F_1{-score}_2$')
# 显示图例
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(project_list))

# 设置横坐标刻度
for ax in axs:
    ax.set_xticks(epoch_list)
plt.tight_layout()
plt.show()

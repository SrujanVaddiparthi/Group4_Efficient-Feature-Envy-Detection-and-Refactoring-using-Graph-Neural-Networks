import os
import time
import subprocess
import re
import matplotlib.pyplot as plt
import statistics

project_list = ['activemq', 'alluxio', 'binnavi', 'kafka', 'realm-java']

encoding_list = [1, 3, 2]
random_seed_list =list(range(5))
repeat_time = 5
t_start = time.time()
for project in project_list:
    for encoding in encoding_list:
        for random_seed in random_seed_list:
            filename = f"RQ2/{project}_{encoding}_{random_seed}.txt"
                
            if os.path.exists(filename):
                print(f"{filename} already exists, skipping...")
                continue

            t_round = time.time()
            command = f"python train.py --project {project} --encoding {encoding} --random_seed {random_seed} --repeat_time {repeat_time}"
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            stdout, _ = process.communicate()

            with open(filename, "w") as file:
                file.write(stdout.decode())
                print(f'Results have been saved to {filename}')
            print('This round time:', time.time() - t_round)

print('Total time:', time.time() - t_start)


def extract_metrics(content, metric_name):
    match = re.search(f"{metric_name}:\s+(\d+\.\d+)%", content)
    if match:
        return float(match.group(1))
    return None

def process_file(filename):
    
    if not os.path.exists(filename):
        return None

    with open(filename, "r") as file:
        content = file.read()

    start_index = content.find("Test set\n") + len("Test set\n")
    results = content[start_index:]

    precision = extract_metrics(results, "Precision2")
    recall = extract_metrics(results, "Recall2")
    f1_score = extract_metrics(results, "F1-Score2")

    return precision, recall, f1_score

precisions = {}
recalls = {}
f1_scores = {}

for project in project_list:
    for encoding in encoding_list:
        precision_list = []
        recall_list = []
        f1_score_list = []
        for random_seed in random_seed_list:
            filename = f"RQ2/{project}_{encoding}_{random_seed}.txt"
            precision, recall, f1_score = process_file(filename)    

            if precision is not None:
                precision_list.append(precision)

            if recall is not None:
                recall_list.append(recall)

            if f1_score is not None:
                f1_score_list.append(f1_score)

        precisions[(project, encoding)] = statistics.mean(precision_list)
        recalls[(project, encoding)] = statistics.mean(recall_list)
        f1_scores[(project, encoding)] = statistics.mean(f1_score_list)

        print(f"Test set results for {project}_{encoding}:")
        print(f"Precision: {precisions[(project, encoding)]}")
        print(f"Recall: {recalls[(project, encoding)]}")
        print(f"F1 Score: {f1_scores[(project, encoding)]}")
        print()

# fig, axs = plt.subplots(3, 1, figsize=(8, 12))  # 创建一个3行1列的子图布局
project_dic = {'activemq': 'ActiveMQ', 'alluxio': 'Alluxio', 'binnavi': 'BinNavi', 'kafka': 'Kafka', 'realm-java': 'Realm-java'}
encoding_dic = {1: 'Position and Semantic', 2: 'Position Only', 3: 'Semantic Only'}

# for i, metric in enumerate([precisions, recalls, f1_scores]):
#     ax = axs[i]  # 获取当前子图对象
#     for encoding in encoding_list:
#         data = [metric[(project, encoding)] for project in project_list]  # 获取当前指标下的数据
#         ax.plot(project_dic.values(), data,  marker='o', label=encoding_dic[encoding])  # 绘制折线图

#     ax.legend(loc='upper left')  # 添加图例.
# plt.rcParams["text.usetex"] = True
# axs[0].set_ylabel(r'$precision_2$')  # 设置y轴标签
# axs[1].set_ylabel(r'$recall_2$')  # 设置y轴标签
# axs[2].set_ylabel(r'$F1{-score}_2$')  # 设置y轴标签
# plt.tight_layout()  # 调整子图布局
# plt.show()  # 显示图表
import numpy as np
import matplotlib.pyplot as plt

# ... existing code ...

fig, axs = plt.subplots(3, 1, figsize=(8, 12))  # Create a 3x1 subplot layout
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
hatches = ['.', '*', 'x', 'o', 'O', ]
for i, metric in enumerate([precisions, recalls, f1_scores]):
    ax = axs[i]  # Get the current subplot object
    x = np.arange(len(project_list))  # Create an array of x values
    width = 0.2  # Width of each bar
    spacing = 0.05  # Spacing between bars for each project

    for j, encoding in enumerate(encoding_list):
        data = [metric[(project, encoding)] for project in project_list]  # Get the data for the current encoding
        ax.bar(x + (j - 1) * (width + spacing), data, width, label=encoding_dic[encoding], 
            edgecolor='black', linewidth=0.8)  # Plot the bar chart with black edges

    ax.set_xticks(x)  # Set the x-axis tick positions
    ax.set_xticklabels(project_dic.values())  # Set the x-axis tick labels with rotation
    # ax.legend(loc='upper left')  # Add legend

    # Add data labels
    # for j, encoding in enumerate(encoding_list):
    #     data = [metric[(project, encoding)] for project in project_list]  # Get the data for the current encoding
    #     for k, d in enumerate(data):
    #         ax.text(x[k] + (j - 1) * (width + spacing), d + 0.5, f'{d:.2f}', ha='center', va='bottom', fontsize=8)

plt.rcParams["text.usetex"] = True

for ax in axs:
    ax.set_ylim(0, 103)
axs[0].set_ylabel('$precision_2$')  # Set y-axis label
axs[1].set_ylabel('$recall_2$')  # Set y-axis label
axs[2].set_ylabel('$F_1{-score}_2$')  # Set y-axis label
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(encoding_list))

plt.tight_layout()  # Adjust subplot layout
plt.show()  # Display the chart

import os
import time
import subprocess
import re
import statistics

project_list = ['activemq', 'alluxio', 'binnavi', 'kafka', 'realm-java']
conv_list = ['GAT', 'GCN', 'Sage']
random_seed_list =list(range(5))
repeat_time = 5
t_start = time.time()
device = 'cuda'
for project in project_list:
    for conv in conv_list:
        for random_seed in random_seed_list:
            filename = f"RQ3/{project}_{conv}_{random_seed}.txt"
                
            if os.path.exists(filename):
                print(f"{filename} already exists, skipping...")
                continue

            t_round = time.time()
            command = f"python train.py --project {project} --conv {conv} --random_seed {random_seed} --repeat_time {repeat_time} --device {device}"
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            stdout, stderr = process.communicate()

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
    for conv in conv_list:
        precision_list = []
        recall_list = []
        f1_score_list = []

        for random_seed in random_seed_list:
            filename = f"RQ3/{project}_{conv}_{random_seed}.txt"
            precision, recall, f1_score = process_file(filename)

            if precision is not None:
                precision_list.append(precision)

            if recall is not None:
                recall_list.append(recall)

            if f1_score is not None:
                f1_score_list.append(f1_score)
        precisions[(project, conv)] = statistics.mean(precision_list)
        recalls[(project, conv)] = statistics.mean(recall_list)
        f1_scores[(project, conv)] = statistics.mean(f1_score_list)

        print(f"Test set results for {project}_{conv}:")
        print(f"Precision: {precisions[(project, conv)]}")
        print(f"Recall: {recalls[(project, conv)]}")
        print(f"F1 Score: {f1_scores[(project, conv)]}")
        print()


from tabulate import tabulate

# 定义表头
headers = ['Project', '$precision_2$', '$recall_2$', '$F1\\text{-score}_2$', '$precision_2$', '$recall_2$', '$F1\\text{-score}_2$', '$precision_2$', '$recall_2$', '$F1\\text{-score}_2$']

# 创建一个空的结果列表
table_data = []

# 遍历每个项目
project_dic = {'binnavi': 'BinNavi', 'activemq': 'ActiveMQ', 'kafka': 'Kafka', 'alluxio': 'Alluxio', 'realm-java': 'Realm-java'}
for project in project_list:
    # 获取每个项目的结果
    gat_precision = precisions[(project, 'GAT')]
    gat_recall = recalls[(project, 'GAT')]
    gat_f1 = f1_scores[(project, 'GAT')]

    gcn_precision = precisions[(project, 'GCN')]
    gcn_recall = recalls[(project, 'GCN')]
    gcn_f1 = f1_scores[(project, 'GCN')]

    sage_precision = precisions[(project, 'Sage')]
    sage_recall = recalls[(project, 'Sage')]
    sage_f1 = f1_scores[(project, 'Sage')]

    max_precision = max([gat_precision, gcn_precision, sage_precision])
    max_recall = max([gat_recall, gcn_recall, sage_recall])
    max_f1 = max([gat_f1, gcn_f1, sage_f1])


    table_data.append([
        project_dic[project],
        r'\textbf{' + '{:.2f}'.format(gat_precision) + r'\%}' if gat_precision == max_precision else '{:.2f}'.format(gat_precision) + r'\%',
        r'\textbf{' + '{:.2f}'.format(gat_recall) + r'\%}' if gat_recall == max_recall else '{:.2f}'.format(gat_recall) + r'\%',
        r'\textbf{' + '{:.2f}'.format(gat_f1) + r'\%}' if gat_f1 == max_f1 else '{:.2f}'.format(gat_f1) + r'\%',
        r'\textbf{' + '{:.2f}'.format(gcn_precision) + r'\%}' if gcn_precision == max_precision else '{:.2f}'.format(gcn_precision) + r'\%',
        r'\textbf{' + '{:.2f}'.format(gcn_recall) + r'\%}' if gcn_recall == max_recall else '{:.2f}'.format(gcn_recall) + r'\%',
        r'\textbf{' + '{:.2f}'.format(gcn_f1) + r'\%}' if gcn_f1 == max_f1 else '{:.2f}'.format(gcn_f1) + r'\%',
        r'\textbf{' + '{:.2f}'.format(sage_precision) + r'\%}' if sage_precision == max_precision else '{:.2f}'.format(sage_precision) + r'\%',
        r'\textbf{' + '{:.2f}'.format(sage_recall) + r'\%}' if sage_recall == max_recall else '{:.2f}'.format(sage_recall) + r'\%',
        r'\textbf{' + '{:.2f}'.format(sage_f1) + r'\%}' if sage_f1 == max_f1 else '{:.2f}'.format(sage_f1) + r'\%'
    ])

gat_average_precision = statistics.mean([precisions[(project, 'GAT')] for project in project_list])
gat_average_recall = statistics.mean([recalls[(project, 'GAT')] for project in project_list])
gat_average_f1 = statistics.mean([f1_scores[(project, 'GAT')] for project in project_list])

gcn_average_precision = statistics.mean([precisions[(project, 'GCN')] for project in project_list])
gcn_average_recall = statistics.mean([recalls[(project, 'GCN')] for project in project_list])
gcn_average_f1 = statistics.mean([f1_scores[(project, 'GCN')] for project in project_list])

sage_average_precision = statistics.mean([precisions[(project, 'Sage')] for project in project_list])
sage_average_recall = statistics.mean([recalls[(project, 'Sage')] for project in project_list])
sage_average_f1 = statistics.mean([f1_scores[(project, 'Sage')] for project in project_list])

max_precision = max([gat_average_precision, gcn_average_precision, sage_average_precision])
max_recall = max([gat_average_recall, gcn_average_recall, sage_average_recall])
max_f1 = max([gat_average_f1, gat_average_f1, sage_average_f1])

# Add average row to table data
table_data.append([
    r'\textbf{Average}',
    r'\textbf{' + '{:.2f}'.format(gat_average_precision) + r'\%}' if gat_average_precision == max_precision else '{:.2f}'.format(gat_average_precision) + r'\%',
    r'\textbf{' + '{:.2f}'.format(gat_average_recall) + r'\%}' if gat_average_recall == max_recall else '{:.2f}'.format(gat_average_recall) + r'\%',
    r'\textbf{' + '{:.2f}'.format(gat_average_f1) + r'\%}' if gat_average_f1 == max_f1 else '{:.2f}'.format(gat_average_f1) + r'\%',
    r'\textbf{' + '{:.2f}'.format(gcn_average_precision) + r'\%}' if gcn_average_precision == max_precision else '{:.2f}'.format(gcn_average_precision) + r'\%',
    r'\textbf{' + '{:.2f}'.format(gcn_average_recall) + r'\%}' if gcn_average_recall == max_recall else '{:.2f}'.format(gcn_average_recall) + r'\%',
    r'\textbf{' + '{:.2f}'.format(gcn_average_f1) + r'\%}' if gcn_average_f1 == max_f1 else '{:.2f}'.format(gcn_average_f1) + r'\%',
    r'\textbf{' + '{:.2f}'.format(sage_average_precision) + r'\%}' if sage_average_precision == max_precision else '{:.2f}'.format(sage_average_precision) + r'\%',
    r'\textbf{' + '{:.2f}'.format(sage_average_recall) + r'\%}' if sage_average_recall == max_recall else '{:.2f}'.format(sage_average_recall) + r'\%',
    r'\textbf{' + '{:.2f}'.format(sage_average_f1) + r'\%}' if sage_average_f1 == max_f1 else '{:.2f}'.format(sage_average_f1) + r'\%'
])

# 将表格数据转换为LaTeX表格
latex_table = tabulate(table_data, headers, tablefmt='latex')
latex_table = latex_table.replace('\\textbackslash{}', '').replace('\\}', '}').replace('\\{', '{').replace('text', '\\text').replace('\\$', '$').replace('\\_','_')
# 打印LaTeX表格
print(latex_table)



import numpy as np
import matplotlib.pyplot as plt

# ... existing code ...

fig, axs = plt.subplots(3, 1, figsize=(8, 12))  # Create a 3x1 subplot layout
conv_dic = {'GAT':'GAT', 'GCN':'GCN', 'Sage':'GraphSAGE'}
for i, metric in enumerate([precisions, recalls, f1_scores]):
    ax = axs[i]  # Get the current subplot object
    x = np.arange(len(project_list))  # Create an array of x values
    width = 0.2  # Width of each bar
    spacing = 0.05  # Spacing between bars for each project

    for j, conv in enumerate(conv_list):
        data = [metric[(project, conv)] for project in project_list]  # Get the data for the current encoding
        ax.bar(x + (j - 1) * (width + spacing), data, width, label=conv_dic[conv], edgecolor='black', linewidth=0.8)  # Plot the bar chart with black edges

    ax.set_xticks(x)  # Set the x-axis tick positions
    ax.set_xticklabels(project_dic.values())  # Set the x-axis tick labels with rotation
    # ax.legend(loc='upper left')  # Add legend

    # Add data labels
    # for j, conv in enumerate(conv_list):
    #     data = [metric[(project, conv)] for project in project_list]  # Get the data for the current encoding
    #     for k, d in enumerate(data):
    #         ax.text(x[k] + (j - 1) * (width + spacing), d + 0.5, f'{d:.2f}', ha='center', va='bottom', fontsize=8)

plt.rcParams["text.usetex"] = True
for ax in axs:
    ax.set_ylim(0, 103)
axs[0].set_ylabel(r'$precision_2$')  # Set y-axis label
axs[1].set_ylabel(r'$recall_2$')  # Set y-axis label
axs[2].set_ylabel(r'$F_1{-score}_2$')  # Set y-axis label
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(conv_list))

plt.tight_layout()  # Adjust subplot layout
plt.show()  # Display the chart

import os
import time
import subprocess
import re
import matplotlib.pyplot as plt
import statistics

project_list = ['activemq', 'alluxio', 'binnavi', 'kafka', 'realm-java']
random_seed_list = list(range(5))
repeat_time = 1
t_start = time.time()
device = 'cuda'
weights = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
for project in project_list:
    for weight in weights:
        for random_seed in random_seed_list:
            filename = f"RQ1/{project}_{random_seed}_{weight}.txt"
                
            if os.path.exists(filename):
                print(f"{filename} already exists, skipping...")
                continue
            t_round = time.time()
            command = f"python train.py --project {project} --random_seed {random_seed} --repeat_time {repeat_time} --device {device} --weight {weight}"
                
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            stdout, stderr = process.communicate()

            with open(filename, "w") as file:
                file.write(stdout.decode())
                print(f'Results have been saved to {filename}')
            print('This round time:', time.time() - t_round)

print('Total time:', time.time() - t_start)


def process_file(filename):
    def extract_metrics(content, metric_name):
        match = re.search(f"{metric_name}:\s+(\d+\.\d+)%", content)
        if match:
            return float(match.group(1))
        return None
    
    if not os.path.exists(filename):
        return None

    with open(filename, "r") as file:
        content = file.read()

    start_index = content.find("Test set\n") + len("Test set\n")
    results = content[start_index:]

    f1_score1 = extract_metrics(results, "F1-Score1")

    return f1_score1

f1_scores = {}

for project in project_list:
    for weight in weights:
        f1_score1_list = []
        for random_seed in random_seed_list:
            filename = f"RQ1/{project}_{random_seed}_{weight}.txt"
            f1_score1 = process_file(filename)
            
            if f1_score1 is not None:
                f1_score1_list.append(f1_score1)

        f1_scores[(project, weight)] = statistics.mean(f1_score1_list)

        print(f"Test set results for {project}-{weight}:")

        print("Average F1_score1: {:.2f}".format(f1_scores[(project, weight)]))
        print()


import matplotlib.pyplot as plt
import pandas as pd


dic = {}
for project in project_list:
    dic[project] = [f1_scores[(project, weight)] for weight in weights]


# 绘制折线图
plt.figure(figsize=(10, 6))  # 设置图形大小
x = range(1, len(weights) + 1)  # x轴标签
markers = ['o', 's', 'D', 'v', 'x']  # 不同点形状
linestyles = ['-', '--', '-.', ':', '-']  # 不同线条形状
project_dic = {'binnavi': 'BinNavi', 'activemq': 'ActiveMQ', 'kafka': 'Kafka', 'alluxio': 'Alluxio', 'realm-java': 'Realm-java'}
for i, project in enumerate(project_list):
    plt.plot(x, dic[project], linestyle=linestyles[i % len(linestyles)], marker=markers[i % len(markers)], label=project_dic[project])  # 绘制折线图
plt.xlabel('$\lambda$')  # 设置x轴标签
plt.ylabel('$F_1-score_1$')  # 设置y轴标签
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格
plt.xticks(x, ['1E-7', '5E-7', '1E-6', '5E-6', '1E-5', '5E-5', '1E-4'])  # 设置x轴刻度
plt.ylim(60, 102)
plt.show()  # 显示图形

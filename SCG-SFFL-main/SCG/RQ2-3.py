import os
import time
import subprocess
import re
import statistics

project_list = ['activemq', 'alluxio', 'binnavi', 'kafka', 'realm-java']
random_seed_list = list(range(5))
repeat_time = 1
t_start = time.time()
device = 'cuda'
for project in project_list:
    for random_seed in random_seed_list:
        filename = f"RQ2-3/{project}_{random_seed}.txt"
            
        if os.path.exists(filename):
            print(f"{filename} already exists, skipping...")
            continue
        t_round = time.time()
        command = f"python train.py --project {project} --random_seed {random_seed} --repeat_time {repeat_time} --device {device}"
            
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

    precision1 = extract_metrics(results, "Precision1")
    recall1 = extract_metrics(results, "Recall1")
    f1_score1 = extract_metrics(results, "F1-Score1")

    precision2 = extract_metrics(results, "Precision2")
    recall2 = extract_metrics(results, "Recall2")
    f1_score2 = extract_metrics(results, "F1-Score2")

    acc_liu = recall2 / recall1 * 100
    return precision1, recall1, f1_score1, precision2, recall2, f1_score2, acc_liu

precisions = {}
recalls = {}
f1_scores = {}
accs = {}
for project in project_list:
    precision1_list = []
    recall1_list = []
    f1_score1_list = []
    precision2_list = []
    recall2_list = []
    f1_score2_list = []
    acc_liu_list = []
    for random_seed in random_seed_list:
        filename = f"RQ2-3/{project}_{random_seed}.txt"
        precision1, recall1, f1_score1, precision2, recall2, f1_score2, acc_liu = process_file(filename)

        if precision1 is not None:
            precision1_list.append(precision1)

        if recall1 is not None:
            recall1_list.append(recall1)

        if f1_score1 is not None:
            f1_score1_list.append(f1_score1)
        
        if precision2 is not None:
            precision2_list.append(precision2)

        if recall2 is not None:
            recall2_list.append(recall2)

        if f1_score2 is not None:
            f1_score2_list.append(f1_score2)
        
        if acc_liu is not None:
            acc_liu_list.append(acc_liu)

    precisions[(project, '1')] = statistics.mean(precision1_list)
    recalls[(project, '1')] = statistics.mean(recall1_list)
    f1_scores[(project, '1')] = statistics.mean(f1_score1_list)

    precisions[(project, '2')] = statistics.mean(precision2_list)
    recalls[(project, '2')] = statistics.mean(recall2_list)
    f1_scores[(project, '2')] = statistics.mean(f1_score2_list)

    accs[project] = statistics.mean(acc_liu_list)

    print(f"Test set results for {project}:")
    print("Average Precision1: {:.2f}".format(precisions[(project, '1')]))
    print("Average Recall1: {:.2f}".format(recalls[(project, '1')]))
    print("Average F1_score1: {:.2f}".format(f1_scores[(project, '1')]))

    print("Average Precision2: {:.2f}".format(precisions[(project, '2')]))
    print("Average Recall2: {:.2f}".format(recalls[(project, '2')]))
    print("Average F1_score2: {:.2f}".format(f1_scores[(project, '2')]))

    print("Average ACC_liu: {:.2f}".format(accs[project]))
    print()


from tabulate import tabulate

# 定义表头
headers = ['Project', '$precision_1$', '$recall_1$', '$F1\\text{-score}_1$', '$liu_acc$', '$precision_2$', '$recall_2$', '$F1\\text{-score}_2$']

# 创建一个空的结果列表
table_data = []

# 遍历每个项目
project_dic = {'binnavi': 'BinNavi', 'activemq': 'ActiveMQ', 'kafka': 'Kafka', 'alluxio': 'Alluxio', 'realm-java': 'Realm-java'}
for project in project_list:
    
    precision1, recall1, f1_score1 = precisions[(project, '1')], recalls[(project, '1')], f1_scores[(project, '1')]
    precision2, recall2, f1_score2 = precisions[(project, '2')], recalls[(project, '2')], f1_scores[(project, '2')]
    acc = accs[project]
    table_data.append([
        project_dic[project],
        '{:.2f}'.format(precision1) + r'\%',
        '{:.2f}'.format(recall1) + r'\%',
        '{:.2f}'.format(f1_score1) + r'\%',
        '{:.2f}'.format(acc) + r'\%',
        '{:.2f}'.format(precision2) + r'\%',
        '{:.2f}'.format(recall2) + r'\%',
        '{:.2f}'.format(f1_score2) + r'\%',
    ])

average_precision1 = statistics.mean([precisions[(project, '1')] for project in project_list])
average_recall1 = statistics.mean([recalls[(project, '1')] for project in project_list])
average_f1_score1 = statistics.mean([f1_scores[(project, '1')] for project in project_list])

average_acc_liu = statistics.mean([accs[project] for project in project_list])

average_precision2 = statistics.mean([precisions[(project, '2')] for project in project_list])
average_recall2 = statistics.mean([recalls[(project, '2')] for project in project_list])
average_f1_score2 = statistics.mean([f1_scores[(project, '2')] for project in project_list])


# Add average row to table data
table_data.append([
    r'\textbf{Average}',
    '{:.2f}'.format(average_precision1) + r'\%',
    '{:.2f}'.format(average_recall1) + r'\%',
    '{:.2f}'.format(average_f1_score1) + r'\%',
    '{:.2f}'.format(average_acc_liu) + r'\%',
    '{:.2f}'.format(average_precision2) + r'\%',
    '{:.2f}'.format(average_recall2) + r'\%',
    '{:.2f}'.format(average_f1_score2) + r'\%',
])

# 将表格数据转换为LaTeX表格
latex_table = tabulate(table_data, headers, tablefmt='latex')
latex_table = latex_table.replace('\\textbackslash{}', '').replace('\\}', '}').replace('\\{', '{').replace('text', '\\text').replace('\\$', '$').replace('\\_','_')
# 打印LaTeX表格
print(latex_table)#14194
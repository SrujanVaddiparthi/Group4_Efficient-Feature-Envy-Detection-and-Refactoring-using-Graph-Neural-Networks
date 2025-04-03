import os
import time
import subprocess
import re
import statistics

project_list = ['activemq', 'alluxio', 'binnavi', 'kafka', 'realm-java']
random_seed_list = list(range(5))
repeat_time =5
t_start = time.time()
device = 'cuda'
for project in project_list:
    for random_seed in random_seed_list:
        filename = f"RQ4/SFFL/{project}_{random_seed}.txt"
            
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



def calculate_metrics(folder, project_list, random_seed_list):
    precisions = {}
    recalls = {}
    f1_scores = {}
    accs = {}
    for project in project_list:
        precision1_list = []
        recall1_list = []
        f1_score1_list = []
        acc_list = []
        precision2_list = []
        recall2_list = []
        f1_score2_list = []
        
        for random_seed in random_seed_list:
            filename = f"RQ4/{folder}/{project}_{random_seed}.txt"
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
                acc_list.append(acc_liu)

        precisions[(project, '1')] = statistics.mean(precision1_list)
        recalls[(project, '1')] = statistics.mean(recall1_list)
        f1_scores[(project, '1')] = statistics.mean(f1_score1_list)

        accs[project] = statistics.mean(acc_list)

        precisions[(project, '2')] = statistics.mean(precision2_list)
        recalls[(project, '2')] = statistics.mean(recall2_list)
        f1_scores[(project, '2')] = statistics.mean(f1_score2_list)

        print(f"Test set results for {project}:")
        print("Average Precision1: {:.2f}".format(precisions[(project, '1')]))
        print("Average Recall1: {:.2f}".format(recalls[(project, '1')]))
        print("Average F1_score1: {:.2f}".format(f1_scores[(project, '1')]))

        print("Average ACC_liu: {:.2f}".format(accs[project]))

        print("Average Precision2: {:.2f}".format(precisions[(project, '2')]))
        print("Average Recall2: {:.2f}".format(recalls[(project, '2')]))
        print("Average F1_score2: {:.2f}".format(f1_scores[(project, '2')]))
        print()

    return precisions, recalls, f1_scores, accs

SCG_precisions, SCG_recalls, SCG_f1_scores, SCG_accs = calculate_metrics('SCG', project_list, random_seed_list)
SFFL_precisions, SFFL_recalls, SFFL_f1_scores, SFFL_accs = calculate_metrics('SFFL', project_list, random_seed_list)

from tabulate import tabulate

# 定义表头
headers = [
    'Project', 'Approach',
    r'$\textit{precision}_1$', r'$\textit{recall}_1$', r'$\textit{F}_1\text{-score}_1$', r'$\textit{accuracy}$'  
    r'$\textit{precision}_2$', r'$\textit{recall}_2$', r'$\textit{F}_1\text{-score}_2$',
]

# 创建一个空的结果列表
table_data = []

# 遍历每个项目
project_dic = {'binnavi': 'BinNavi', 'activemq': 'ActiveMQ', 'kafka': 'Kafka', 'alluxio': 'Alluxio', 'realm-java': 'Realm-java'}
for project in project_list:
    
    SCG_precision1, SCG_recall1, SCG_f1_score1 = SCG_precisions[(project, '1')], SCG_recalls[(project, '1')], SCG_f1_scores[(project, '1')]
    SCG_precision2, SCG_recall2, SCG_f1_score2 = SCG_precisions[(project, '2')], SCG_recalls[(project, '2')], SCG_f1_scores[(project, '2')]

    SFFL_precision1, SFFL_recall1, SFFL_f1_score1 = SFFL_precisions[(project, '1')], SFFL_recalls[(project, '1')], SFFL_f1_scores[(project, '1')]
    SFFL_precision2, SFFL_recall2, SFFL_f1_score2 = SFFL_precisions[(project, '2')], SFFL_recalls[(project, '2')], SFFL_f1_scores[(project, '2')]

    SCG_acc = SCG_accs[project]
    SFFL_acc = SFFL_accs[project]
    
    max_precision1 = max(SFFL_precision1, SCG_precision1)
    max_recall1 = max(SFFL_recall1, SCG_recall1)
    max_f1_score1 = max(SFFL_f1_score1, SCG_f1_score1)

    max_precision2 = max(SFFL_precision2, SCG_precision2)
    max_recall2 = max(SFFL_recall2, SCG_recall2)
    max_f1_score2 = max(SFFL_f1_score2, SCG_f1_score2)

    max_acc = max(SFFL_acc, SCG_acc)

    table_data.append([
        r'\multirow{2}{*}{' + project_dic[project] + r'}',
        r'SCG',
        r'\textbf{' + '{:.2f}'.format(SCG_precision1) + r'\%}' if SCG_precision1 == max_precision1 else '{:.2f}'.format(SCG_precision1) + r'\%',
        r'\textbf{' + '{:.2f}'.format(SCG_recall1) + r'\%}' if SCG_recall1 == max_recall1 else '{:.2f}'.format(SCG_recall1) + r'\%',
        r'\textbf{' + '{:.2f}'.format(SCG_f1_score1) + r'\%}' if SCG_f1_score1 == max_f1_score1 else '{:.2f}'.format(SCG_f1_score1) + r'\%',
        r'\textbf{' + '{:.2f}'.format(SCG_acc) + r'\%}' if SCG_acc == max_acc else '{:.2f}'.format(SCG_acc) + r'\%',
        r'\textbf{' + '{:.2f}'.format(SCG_precision2) + r'\%}' if SCG_precision2 == max_precision2 else '{:.2f}'.format(SCG_precision2) + r'\%',
        r'\textbf{' + '{:.2f}'.format(SCG_recall2) + r'\%}' if SCG_recall2 == max_recall2 else '{:.2f}'.format(SCG_recall2) + r'\%',
        r'\textbf{' + '{:.2f}'.format(SCG_f1_score2) + r'\%}' if SCG_f1_score2 == max_f1_score2 else '{:.2f}'.format(SCG_f1_score2) + r'\%',
    ])

    table_data.append([
        r' ',
        r'SFFL',
        r'\textbf{' + '{:.2f}'.format(SFFL_precision1) + r'\%}' if SFFL_precision1 == max_precision1 else '{:.2f}'.format(SFFL_precision1) + r'\%',
        r'\textbf{' + '{:.2f}'.format(SFFL_recall1) + r'\%}' if SFFL_recall1 == max_recall1 else '{:.2f}'.format(SFFL_recall1) + r'\%',
        r'\textbf{' + '{:.2f}'.format(SFFL_f1_score1) + r'\%}' if SFFL_f1_score1 == max_f1_score1 else '{:.2f}'.format(SFFL_f1_score1) + r'\%',
        r'\textbf{' + '{:.2f}'.format(SFFL_acc) + r'\%}' if SFFL_acc == max_acc else '{:.2f}'.format(SFFL_acc) + r'\%',
        r'\textbf{' + '{:.2f}'.format(SFFL_precision2) + r'\%}' if SFFL_precision2 == max_precision2 else '{:.2f}'.format(SFFL_precision2) + r'\%',
        r'\textbf{' + '{:.2f}'.format(SFFL_recall2) + r'\%}' if SFFL_recall2 == max_recall2 else '{:.2f}'.format(SFFL_recall2) + r'\%',
        r'\textbf{' + '{:.2f}'.format(SFFL_f1_score2) + r'\%}' if SFFL_f1_score2 == max_f1_score2 else '{:.2f}'.format(SFFL_f1_score2) + r'\%',
    ])

SCG_average_precision1 = statistics.mean([SCG_precisions[(project, '1')] for project in project_list])
SCG_average_recall1 = statistics.mean([SCG_recalls[(project, '1')] for project in project_list])
SCG_average_f1_score1 = statistics.mean([SCG_f1_scores[(project, '1')] for project in project_list])
SCG_average_acc = statistics.mean([SCG_accs[project] for project in project_list])
SCG_average_precision2 = statistics.mean([SCG_precisions[(project, '2')] for project in project_list])
SCG_average_recall2 = statistics.mean([SCG_recalls[(project, '2')] for project in project_list])
SCG_average_f1_score2 = statistics.mean([SCG_f1_scores[(project, '2')] for project in project_list])

SFFL_average_precision1 = statistics.mean([SFFL_precisions[(project, '1')] for project in project_list])
SFFL_average_recall1 = statistics.mean([SFFL_recalls[(project, '1')] for project in project_list])
SFFL_average_f1_score1 = statistics.mean([SFFL_f1_scores[(project, '1')] for project in project_list])
SFFL_average_acc = statistics.mean([SFFL_accs[project] for project in project_list])
SFFL_average_precision2 = statistics.mean([SFFL_precisions[(project, '2')] for project in project_list])
SFFL_average_recall2 = statistics.mean([SFFL_recalls[(project, '2')] for project in project_list])
SFFL_average_f1_score2 = statistics.mean([SFFL_f1_scores[(project, '2')] for project in project_list])

max_avg_precision1 = max(SFFL_average_precision1, SCG_average_precision1)
max_avg_recall1 = max(SFFL_average_recall1, SCG_average_recall1)
max_avg_f1_score1 = max(SFFL_average_f1_score1, SCG_average_f1_score1)
max_avg_acc = max(SFFL_average_acc, SCG_average_acc)
max_avg_precision2 = max(SFFL_average_precision2, SCG_average_precision2)
max_avg_recall2 = max(SFFL_average_recall2, SCG_average_recall2)
max_avg_f1_score2 = max(SFFL_average_f1_score2, SCG_average_f1_score2)

# Add average row to table data

table_data.append([
    r'\multirow{2}{*}{\textbf{Average}}',
    r'SCG',
    r'\textbf{' + '{:.2f}'.format(SCG_average_precision1) + r'\%}' if SCG_average_precision1 == max_avg_precision1 else '{:.2f}'.format(SCG_average_precision1) + r'\%',
    r'\textbf{' + '{:.2f}'.format(SCG_average_recall1) + r'\%}' if SCG_average_recall1 == max_avg_recall1 else '{:.2f}'.format(SCG_average_recall1) + r'\%',
    r'\textbf{' + '{:.2f}'.format(SCG_average_f1_score1) + r'\%}' if SCG_average_f1_score1 == max_avg_f1_score1 else '{:.2f}'.format(SCG_average_f1_score1) + r'\%',
    r'\textbf{' + '{:.2f}'.format(SCG_average_acc) + r'\%}' if SCG_average_acc == max_avg_acc else '{:.2f}'.format(SCG_average_acc) + r'\%',
    r'\textbf{' + '{:.2f}'.format(SCG_average_precision2) + r'\%}' if SCG_average_precision2 == max_avg_precision2 else '{:.2f}'.format(SCG_average_precision2) + r'\%',
    r'\textbf{' + '{:.2f}'.format(SCG_average_recall2) + r'\%}' if SCG_average_recall2 == max_avg_recall2 else '{:.2f}'.format(SCG_average_recall2) + r'\%',
    r'\textbf{' + '{:.2f}'.format(SCG_average_f1_score2) + r'\%}' if SCG_average_f1_score2 == max_avg_f1_score2 else '{:.2f}'.format(SCG_average_f1_score2) + r'\%',
])

table_data.append([
    r' ',
    r'SFFL',
    r'\textbf{' + '{:.2f}'.format(SFFL_average_precision1) + r'\%}' if SFFL_average_precision1 == max_avg_precision1 else '{:.2f}'.format(SFFL_average_precision1) + r'\%',
    r'\textbf{' + '{:.2f}'.format(SFFL_average_recall1) + r'\%}' if SFFL_average_recall1 == max_avg_recall1 else '{:.2f}'.format(SFFL_average_recall1) + r'\%',
    r'\textbf{' + '{:.2f}'.format(SFFL_average_f1_score1) + r'\%}' if SFFL_average_f1_score1 == max_avg_f1_score1 else '{:.2f}'.format(SFFL_average_f1_score1) + r'\%',
    r'\textbf{' + '{:.2f}'.format(SFFL_average_acc) + r'\%}' if SFFL_average_acc == max_avg_acc else '{:.2f}'.format(SFFL_average_acc) + r'\%',
    r'\textbf{' + '{:.2f}'.format(SFFL_average_precision2) + r'\%}' if SFFL_average_precision2 == max_avg_precision2 else '{:.2f}'.format(SFFL_average_precision2) + r'\%',
    r'\textbf{' + '{:.2f}'.format(SFFL_average_recall2) + r'\%}' if SFFL_average_recall2 == max_avg_recall2 else '{:.2f}'.format(SFFL_average_recall2) + r'\%',
    r'\textbf{' + '{:.2f}'.format(SFFL_average_f1_score2) + r'\%}' if SFFL_average_f1_score2 == max_avg_f1_score2 else '{:.2f}'.format(SFFL_average_f1_score2) + r'\%',
])


# 将表格数据转换为LaTeX表格
latex_table = tabulate(table_data, headers, tablefmt='latex')
latex_table = latex_table.replace('\\textbackslash{}', '').replace('\\}', '}').replace('\\{', '{').replace('text', '\\text').replace('\\$', '$').replace('\\_','_').replace('multirow', '\\multirow')
# 打印LaTeX表格
print(latex_table)

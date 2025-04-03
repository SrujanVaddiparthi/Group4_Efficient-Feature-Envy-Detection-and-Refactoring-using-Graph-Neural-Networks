import os
import time
import subprocess
import re

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

project_list = ['activemq', 'alluxio', 'binnavi', 'kafka', 'realm-java']
device = 'cuda'
fine_tune_data = [0, 0.01, 0.05, 0.1]

t_start = time.time()
for pretrained_project in project_list:
    fine_tuned_projects = project_list.copy()
    fine_tuned_projects.remove(pretrained_project)
    
    for fine_tuned_project in fine_tuned_projects:
        
        for ftd in fine_tune_data:
            filename = f"RQ5/{pretrained_project}_{fine_tuned_project}_{ftd}.txt"
                
            if os.path.exists(filename):
                if process_file(filename) != None:
                    print(f"{filename} already exists, skipping...")
                    continue
                
            command = f"python fine-tune.py --pretrained_project {pretrained_project} --fine_tuned_project {fine_tuned_project} --fine_tune_data {ftd}"
                
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            stdout, stderr = process.communicate()

            with open(filename, "w") as file:
                file.write(stdout.decode())
                print(f'Results have been saved to {filename}')

print('Total time:', time.time() - t_start)

f1_scores = {}

for pretrained_project in project_list:
    fine_tuned_projects = project_list.copy()
    fine_tuned_projects.remove(pretrained_project)
    for ftd in fine_tune_data:
        sum_scores = 0
        for fine_tuned_project in fine_tuned_projects:
        
            filename = f"RQ5/{pretrained_project}_{fine_tuned_project}_{ftd}.txt"
            f1_score1 = process_file(filename)
            
            f1_scores[(pretrained_project, fine_tuned_project, ftd)] = f1_score1

            sum_scores += f1_score1
    
            print(f"Test set results for {pretrained_project}-{fine_tuned_project}-{ftd}:")

            print("Average F1_score1: {:.2f}".format(f1_score1))
            print()

        avg_scores = sum_scores / len(fine_tuned_projects)
        f1_scores[(pretrained_project, 'Average', ftd)] = avg_scores
    
from tabulate import tabulate

# 定义表头
headers = ['Tested On', 'Fine-tune', 'ActiveMQ', 'Alluxio', 'BinNavi', 'Kafka', 'Realm-java']

# 创建一个空的结果列表
table_data = []
project_dic = {'binnavi': 'BinNavi', 'activemq': 'ActiveMQ', 'kafka': 'Kafka', 'alluxio': 'Alluxio', 'realm-java': 'Realm-java'}
fine_tune_dic = {0: 'None', 0.01: '1%', 0.05: '5%', 0.1: '10%'}
# 遍历每个项目


for fine_tuned_project in project_list:

    for ftd in fine_tune_data:
        max_data = max([f1_scores[(pretrained_project, fine_tuned_project, ftd)] for pretrained_project in project_list if pretrained_project != fine_tuned_project])
        if ftd == 0:
            row = [f'\multirow{{{len(fine_tune_dic)}}}{{*}}{{{project_dic[fine_tuned_project]}}}', fine_tune_dic[ftd]]
        else:
            row = ['', fine_tune_dic[ftd]]
        for pretrained_project in project_list:
            if pretrained_project == fine_tuned_project:
                row.append('-')
            elif f1_scores[(pretrained_project, fine_tuned_project, ftd)] == max_data:
                row.append(
                    '\\textbf{{{:.2f}}}'.format(f1_scores[(pretrained_project, fine_tuned_project, ftd)]) + '\%',
                )
            else:
                row.append(
                    '{:.2f}'.format(f1_scores[(pretrained_project, fine_tuned_project, ftd)]) + '\%',
                )
        table_data.append(row)

# Add average rows to table data
for ftd in fine_tune_data:
    max_data = max([f1_scores[(pretrained_project, 'Average', ftd)] for pretrained_project in project_list])
    if ftd == 0:
        row = [f'\multirow{{{len(fine_tune_dic)}}}{{*}}{{\\textbf{{Average}}}}', fine_tune_dic[ftd]]
    else:
        row = ['', fine_tune_dic[ftd]]
    for pretrained_project in project_list:
        if f1_scores[(pretrained_project, 'Average', ftd)] == max_data:
            row.append(
                '\\textbf{{{:.2f}}}'.format(f1_scores[(pretrained_project, 'Average', ftd)]) + '\%',
            )
        else:
            row.append(
                '{:.2f}'.format(f1_scores[(pretrained_project, 'Average', ftd)]) + '\%',
            )
    table_data.append(row)


# 将表格数据转换为LaTeX表格
latex_table = tabulate(table_data, headers, tablefmt='latex')
latex_table = latex_table.replace('\\textbackslash{}', '').replace('\\}', '}').replace('\\{', '{').replace('text', '\\text').replace('\\$', '$').replace('\\_','_').replace('multirow', '\\multirow')
# 打印LaTeX表格
print(latex_table)
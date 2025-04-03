import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def process_file(filename, metric_names):
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

    f1_score1 = extract_metrics(results, metric_names)

    return f1_score1

project_list = ['activemq', 'alluxio', 'binnavi', 'kafka', 'realm-java']
fine_tune_data = [0, 0.01, 0.05, 0.1]
approaches = ['SCG', 'SFFL']
score_types = ['F1-score1', 'F1-score2']
f1_scores = {}
for approach in approaches:
    for pretrained_project in project_list:
        fine_tuned_projects = project_list.copy()
        fine_tuned_projects.remove(pretrained_project)
        for ftd in fine_tune_data:
            sum_scores1 = 0
            sum_scores2 = 0
            for fine_tuned_project in fine_tuned_projects:
                
                    filename = f"{approach}/RQ5/{pretrained_project}_{fine_tuned_project}_{ftd}.txt"
                    f1_score1 = process_file(filename, "F1-Score1")
                    f1_score2 = process_file(filename, "F1-Score2")

                    sum_scores1 += f1_score1 if f1_score1 != None else 0
                    sum_scores2 += f1_score2 if f1_score2 != None else 0

            avg_scores1 = sum_scores1 / len(fine_tuned_projects)
            avg_scores2 = sum_scores2 / len(fine_tuned_projects)
            f1_scores[(pretrained_project, ftd, approach, 'F1-score1')] = avg_scores1
            f1_scores[(pretrained_project, ftd, approach, 'F1-score2')] = avg_scores2
    

fig = plt.figure(figsize=(12, 12))
gs = gridspec.GridSpec(3, 6, figure=fig)

project_dic = {'activemq': 'ActiveMQ', 'alluxio': 'Alluxio', 'binnavi': 'BinNavi', 'kafka': 'Kafka', 'realm-java': 'Realm-java'}
fine_tune_dic = {0: 'None', 0.01: '1%', 0.05: '5%', 0.1: '10%'}
ax_list = []
# Loop through each ftd and approach
for i, ftd in enumerate(fine_tune_data[:-1]):
    # Get the data for SCG
    avg_scores1_scg = [f1_scores[(pretrained_project, ftd, 'SCG', 'F1-score1')] for pretrained_project in project_list]
    avg_scores2_scg = [f1_scores[(pretrained_project, ftd, 'SCG', 'F1-score2')] for pretrained_project in project_list]

    # Get the data for SFFL
    avg_scores1_sffl = [f1_scores[(pretrained_project, ftd, 'SFFL', 'F1-score1')] for pretrained_project in project_list]
    avg_scores2_sffl = [f1_scores[(pretrained_project, ftd, 'SFFL', 'F1-score2')] for pretrained_project in project_list]

    # Set the x-axis positions for the bars
    x_pos = np.arange(len(project_list))

    # Plot avg_score1 for SCG
    ax = fig.add_subplot(gs[i // 3, 2*(i % 3):2*(i % 3)+2])
    ax.bar(x_pos - 0.2, avg_scores1_scg, width=0.4, align='center', label='SCG')
    # Plot avg_score1 for SFFL
    ax.bar(x_pos + 0.2, avg_scores1_sffl, width=0.4, align='center', label='SFFL')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(project_dic.values()), rotation=45)
    ax.legend()
    ax_list.append(ax)

    # Plot avg_score2 for SCG
    ax = fig.add_subplot(gs[i // 3 + 1, 2*(i % 3):2*(i % 3)+2])
    ax.bar(x_pos - 0.2, avg_scores2_scg, width=0.4, align='center', label='SCG')
    # Plot avg_score2 for SFFL
    ax.bar(x_pos + 0.2, avg_scores2_sffl, width=0.4, align='center', label='SFFL')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(project_dic.values()), rotation=45)
    ax.legend()
    ax_list.append(ax)
    
# Get the data for SCG
avg_scores1_scg = [f1_scores[(pretrained_project, fine_tune_data[-1], 'SCG', 'F1-score1')] for pretrained_project in project_list]
avg_scores2_scg = [f1_scores[(pretrained_project, fine_tune_data[-1], 'SCG', 'F1-score2')] for pretrained_project in project_list]

# Get the data for SFFL
avg_scores1_sffl = [f1_scores[(pretrained_project, fine_tune_data[-1], 'SFFL', 'F1-score1')] for pretrained_project in project_list]
avg_scores2_sffl = [f1_scores[(pretrained_project, fine_tune_data[-1], 'SFFL', 'F1-score2')] for pretrained_project in project_list]

# Set the x-axis positions for the bars
x_pos = np.arange(len(project_list))

# Plot avg_score1 for SCG
ax = fig.add_subplot(gs[2, 1:3])
ax.bar(x_pos - 0.2, avg_scores1_scg, width=0.4, align='center', label='SCG')
# Plot avg_score1 for SFFL
ax.bar(x_pos + 0.2, avg_scores1_sffl, width=0.4, align='center', label='SFFL')
ax.set_xticks(x_pos)
ax.set_xticklabels(list(project_dic.values()), rotation=45)
ax.legend()
ax_list.append(ax)


# Plot avg_score2 for SCG
ax = fig.add_subplot(gs[2, 3:5])
ax.bar(x_pos - 0.2, avg_scores2_scg, width=0.4, align='center', label='SCG')
# Plot avg_score2 for SFFL
ax.bar(x_pos + 0.2, avg_scores2_sffl, width=0.4, align='center', label='SFFL')
ax.set_xticks(x_pos)
ax.set_xticklabels(list(project_dic.values()), rotation=45)
ax.legend()
ax_list.append(ax)


# fig.delaxes(fig.axes[-1])

for ax in ax_list:
    ax.set_ylim(0, 100)
        
plt.rcParams["text.usetex"] = True
ax_list[0].set_title(f'Average $F_1{{-score}}_1$ (FTD={fine_tune_dic[fine_tune_data[0]]})')
ax_list[1].set_title(f'Average $F_1{{-score}}_2$ (FTD={fine_tune_dic[fine_tune_data[0]]})')

ax_list[2].set_title(f'Average $F_1{{-score}}_1$ (FTD={fine_tune_dic[fine_tune_data[1]]})')
ax_list[3].set_title(f'Average $F_1{{-score}}_2$ (FTD={fine_tune_dic[fine_tune_data[1]]})')

ax_list[4].set_title(f'Average $F_1{{-score}}_1$ (FTD={fine_tune_dic[fine_tune_data[2]]})')
ax_list[5].set_title(f'Average $F_1{{-score}}_2$ (FTD={fine_tune_dic[fine_tune_data[2]]})')

ax_list[6].set_title(f'Average $F_1{{-score}}_1$ (FTD={fine_tune_dic[fine_tune_data[3]]})')
ax_list[7].set_title(f'Average $F_1{{-score}}_2$ (FTD={fine_tune_dic[fine_tune_data[3]]})')

# Set the overall title and adjust the spacing
fig.tight_layout(rect=[0, 0, 1, 0.95])

# Show the plot
plt.show()


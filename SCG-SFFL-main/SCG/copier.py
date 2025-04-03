import shutil
import os


src_dir = 'within_project'
dst_dir1 = 'RQ1'

random_seed_list = list(range(5))
project_list = ['binnavi', 'activemq', 'kafka', 'alluxio', 'realm-java']
weight = 1e-6

for project in project_list:
    for random_seed in random_seed_list:
        src_file = os.path.join(src_dir, f'{project}_{random_seed}.txt')
        dst_file1 = os.path.join(dst_dir1, f'{project}_{random_seed}_{weight}.txt')
        
        shutil.copy2(src_file, dst_file1)

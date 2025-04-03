import shutil
import os


src_dir = 'RQ1'
dst_dir1 = 'RQ2'
dst_dir2 = 'RQ3'
dst_dir3 = 'RQ4/Ours'

hidden_dim = 256
epoch = 300
random_seed_list = list(range(5))
project_list = ['binnavi', 'activemq', 'kafka', 'alluxio', 'realm-java']
encoding = 1
conv = 'GAT'

for project in project_list:
    for random_seed in random_seed_list:
        src_file = os.path.join(src_dir, f'{project}_{hidden_dim}_{epoch}_{random_seed}.txt')
        dst_file1 = os.path.join(dst_dir1, f'{project}_{encoding}_{random_seed}.txt')
        dst_file2 = os.path.join(dst_dir2, f'{project}_{conv}_{random_seed}.txt')
        dst_file3 = os.path.join(dst_dir3, f'{project}_{random_seed}.txt')
        shutil.copy2(src_file, dst_file1)
        shutil.copy2(src_file, dst_file2)
        shutil.copy2(src_file, dst_file3)
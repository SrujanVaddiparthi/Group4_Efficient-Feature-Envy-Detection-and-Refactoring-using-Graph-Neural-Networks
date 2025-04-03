import pandas as pd

project_list = ['activemq', 'alluxio', 'binnavi', 'kafka', 'realm-java']

for project in project_list:

    df1 = pd.read_csv(f'data/{project}/ground_truth.csv')
    df2 = pd.read_csv(f'data/{project}/classInfo.csv')

    method_num = len(df1)
    class_num = len(df2)
    smell_num = df1['label'].sum()

    print(f'{class_num} & {method_num} & {smell_num}')

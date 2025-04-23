import pandas as pd
import os

#the imbalance project
project = 'activemq' 

#path to the CSV file inside scg-sffl-main/scg/data/
data_path = "/Users/wangtiles/Desktop/DSCI644/Group4_Efficient-Feature-Envy-Detection-and-Refactoring-using-Graph-Neural-Networks/SCG-SFFL-main/SCG/data/activemq/ground_truth.csv"
#load it
df = pd.read_csv(data_path)

#count label values
smelly_count = (df['label'] == 1).sum()
normal_count = (df['label'] == 0).sum()
total = len(df)

#print distribution
print(f"Project: {project}")
print(f"Total methods: {total}")
print(f"Smelly methods (label=1): {smelly_count}")
print(f"Normal methods (label=0): {normal_count}")
print(f"Imbalance ratio (smelly/normal): {smelly_count / normal_count:.4f}")


"""that imbalance ratio of about 3.9% clearly shows heavy class imbalance, which is exactly the kind of limitation we want to address."""
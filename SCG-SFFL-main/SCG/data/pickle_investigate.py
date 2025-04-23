# import pickle

# # file_path = '/Users/wangtiles/Desktop/DSCI644/Group4_Efficient-Feature-Envy-Detection-and-Refactoring-using-Graph-Neural-Networks/SCG-SFFL-main/SCG/data/activemq/class_tokens.pkl'  # or 'SCG/method_tokens.pkl'
# file_path = '/Users/wangtiles/Desktop/DSCI644/Group4_Efficient-Feature-Envy-Detection-and-Refactoring-using-Graph-Neural-Networks/SCG-SFFL-main/SCG/data/activemq/method_tokens.pkl'
# # Load the .pkl file
# with open(file_path, 'rb') as file:
#     data = pickle.load(file)

# #contents viewed
# print(data)



# import torch

# model = torch.load(
#     '/Users/wangtiles/Desktop/DSCI644/Group4_Efficient-Feature-Envy-Detection-and-Refactoring-using-Graph-Neural-Networks/SCG-SFFL-main/SCG/pretrained_models/activemq.pth',
#     map_location=torch.device('cpu'),
#     weights_only=True  
# )

import torch

try:
    model = torch.load(
        '/Users/wangtiles/Desktop/DSCI644/Group4_Efficient-Feature-Envy-Detection-and-Refactoring-using-Graph-Neural-Networks/SCG-SFFL-main/SCG/pretrained_models/activemq.pth',
        map_location='cpu',
        weights_only=True
    )
except RuntimeError as e:
    print("Failed with weights_only=True, falling back to legacy load")
    model = torch.load(
        '/Users/wangtiles/Desktop/DSCI644/Group4_Efficient-Feature-Envy-Detection-and-Refactoring-using-Graph-Neural-Networks/SCG-SFFL-main/SCG/pretrained_models/activemq.pth',
        map_location='cpu'
    )

print(type(model))  # Should show the model class
print(model)  # Should show the model architecture
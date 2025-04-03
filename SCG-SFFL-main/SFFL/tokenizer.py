import tiktoken # based on python 10 
import pandas as pd
import pickle
import time
import numpy as np

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def get_tokens(text:str):
    token_integers = encoding.encode(text)
    token_bytes = [encoding.decode_single_token_bytes(token) for token in token_integers]
    tokens = [token_byte.decode('utf-8') for token_byte in token_bytes]
    
    return tokens

def main():

    t = time.time()
    projects = ['activemq', 'alluxio', 'binnavi','kafka', 'realm-java']
    total_tokens = []
    for project in projects:
        class_df = pd.read_csv(f'data/{project}/classInfo.csv')
        method_df = pd.read_csv(f'data/{project}/methodInfo.csv')

        class_tokens = class_df['class'].apply(get_tokens)
        method_tokens = method_df['method'].apply(get_tokens)

        count_strings = lambda x: sum(isinstance(i, str) for i in x)
        class_tokens_num = class_tokens.apply(count_strings)
        method_tokens_num = method_tokens.apply(count_strings)
        # print(np.max(class_tokens_num), np.max(method_tokens_num))
        # print(np.min(class_tokens_num), np.min(method_tokens_num))
        print(np.median(class_tokens_num), np.median(method_tokens_num))
        # print(np.mean(class_tokens_num), np.mean(method_tokens_num))
        # Saving the embeddings to a .pkl file
        with open(f'data/{project}/class_tokens.pkl', 'wb') as f:
            pickle.dump(class_tokens.tolist(), f)

        with open(f'data/{project}/method_tokens.pkl', 'wb') as f:
            pickle.dump(method_tokens.tolist(), f)

        total_tokens = total_tokens + class_tokens.tolist() + method_tokens.tolist()
        # # Loading example
        # with open(f'data/{project}/class_tokens.pkl', 'rb') as f:
        #     tokens = pickle.load(f)
        #     print(tokens)

    with open(f'data/total_tokens.pkl', 'wb') as f:
        pickle.dump(total_tokens, f)

    print(time.time() - t)

def test():
    class_name = "MyFirstClass"
    method_name = "myFirstMethod"
    tokens = [get_tokens(class_name), get_tokens(method_name)]
    print(tokens)

if __name__ == '__main__':
    main()



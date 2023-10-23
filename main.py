import tiktoken
from model import trigram_model as tmodel
def tokenize(file_path):
    encoding = tiktoken.get_encoding("cl100k_base")
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read and print each line
        for line in file:
            data.extend(encoding.encode(line))
    return data
if __name__ == "__main__":
    english_file_path = "data/english.txt"
    python_file_path = "data/Python_code_data.txt"

    model_eng = tmodel(limit = 3, model_type = "natural language")
    model_py = tmodel(limit = 3, model_type = "programming language")
    data1 = tokenize(english_file_path)
    data2 = tokenize(python_file_path)
    model_eng.deterministic_train(data1)
    model_py.deterministic_train(data2)
    self_entropy1 = model_eng.calculate_self_entropy()
    avg_accuracy1 = model_eng.calculate_avg_accuracy()
    cross_entropy1 = model_eng.calculate_cross_entropy(data2)

    
    self_entropy2 = model_py.calculate_self_entropy()
    avg_accuracy2 = model_py.calculate_avg_accuracy()
    cross_entropy2 = model_py.calculate_cross_entropy(data1)

    print("self_entropy of model training on english: ", self_entropy1)
    print("self_entropy of model training on python: ", self_entropy2)
    print("cross_entropy of model training on english: ", cross_entropy1)
    print("cross_entropy of model training on python: ", cross_entropy2)
    print("avg accuracy of model training on english: ", avg_accuracy1)
    print("avg accuracy of model training on python: ", avg_accuracy2)
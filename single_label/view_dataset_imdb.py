from datasets import load_dataset

'''
https://huggingface.co/datasets/stanfordnlp/imdb
0: neg
1: pos
'''
# 定义保存 jsonl 文件的函数
def save_dataset_dict_to_jsonl(dataset_dict, output_dir):
    import os
    os.makedirs(output_dir, exist_ok=True)

    for split, dataset in dataset_dict.items():
        # 定义保存路径
        output_path = os.path.join(output_dir, f"{split}.jsonl")
        # 将 Dataset 对象保存为 jsonl 文件
        dataset.to_json(output_path, orient='records', lines=True)
# In[]


raw_datasets = load_dataset(
            'imdb',
            cache_dir=".cache",
        )
# In[]
raw_datasets.save_to_disk("../datasets/imdb")
# In[]
output_directory = '../jsonls/imdb'
save_dataset_dict_to_jsonl(raw_datasets, output_directory)



# In[]
from datasets import load_dataset,load_from_disk



raw_datasets = load_from_disk("../datasets/imdb")


train_test_split = raw_datasets['train'].train_test_split(
    test_size=0.1,
    stratify_by_column='label',
    seed=42,
    shuffle=True)

# 将切分后的数据集合并到原来的 DatasetDict 中
raw_datasets['train'] = train_test_split['train']
raw_datasets['dev'] = train_test_split['test']

# 检查新的 DatasetDict 结构

# In[]
raw_datasets.save_to_disk("../datasets/imdb_train_dev_test")
# In[]
output_directory = '../jsonls/imdb_train_dev_test'
save_dataset_dict_to_jsonl(raw_datasets, output_directory)

from datasets import load_dataset,Dataset,DatasetDict
import json
from collections import defaultdict

# In[]
dict_id_to_label={}
dict_label_to_id={}
dataset_split_dict=defaultdict()

for split in ['test','dev','train']:
    with open(f"/Users/guoxing.lan/projects/datasets/amazon_reviews_multi/en/{split}.jsonl",'r',encoding="UTF-8") as rf:
        temp_dataset_dict=defaultdict(list)
        for line in rf:
            temp_dict=json.loads(line.strip())
            dict_id_to_label[temp_dict['label']]=temp_dict['label_text']
            dict_label_to_id[temp_dict['label_text']]=temp_dict['label']
            temp_dataset_dict['text'].append(temp_dict['text'])
            temp_dataset_dict['label'].append(temp_dict['label'])
        temp_dataset=Dataset.from_dict(temp_dataset_dict)
        dataset_split_dict[split]=temp_dataset




# In[]
full_dataset = DatasetDict({
    'train': dataset_split_dict['train'],
    'test': dataset_split_dict['test'],
    'dev': dataset_split_dict['dev']
})
print(full_dataset)

# In[]
full_dataset.save_to_disk("../datasets/amazon_train_dev_test")

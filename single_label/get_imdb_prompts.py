import json


id_to_label={1:"positive",0:"negative"}
split='dev'
for split in ['train','dev','test']:
    with open(f"jsonls/imdb_train_dev_test/{split}_prompts.jsonl",'w',encoding="UTF-8") as wf:
        with open(f"jsonls/imdb_train_dev_test/{split}.jsonl",'r',encoding="UTF-8") as rf:
            for line in rf:
                temp_dict=json.loads(line.strip())
                text=temp_dict["text"]
                label=temp_dict["label"]
                prompt=f'''You are a text classification expert. Please help me complete a sentiment classification task. 
    The labels are "positive" and "negative," and the text is {text}. Please provide the most likely label without any additional content.
    '''
                temp_dict={
                    "label":id_to_label[label],
                    "prompt":prompt
                }
                wf.write(json.dumps(temp_dict,ensure_ascii=False)+"\n")

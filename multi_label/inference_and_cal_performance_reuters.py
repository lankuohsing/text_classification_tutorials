import os.path
from utils.eval_utils2 import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
from sklearn.metrics import precision_score, recall_score, f1_score
model_name_or_path="/Users/guoxing.lan/projects/github/text_classification_tutorials/outputs/reuter_bsz-2_lr-2e-05_len-128"


input_data_path="/Users/guoxing.lan/projects/datasets/text_classification/reuters_jsonls/test.jsonl"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

batch_size=4



class MyModel:
    def __init__(self,model_name_or_path, tokenizer_name_or_path=None, device='cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.device=device
        self.model = self.model.to(self.device)

    def batch_inference(self, list_text):
        inputs_batch = self.tokenizer(
            list_text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs_batch)
        pred_probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # 获取每个元素大于 0.5 的布尔矩阵
        mask = pred_probs > 0.01
        # 使用 nonzero 获取大于 0.5 的元素的下标，返回的是二维的 [行索引, 列索引] 的tensor
        indices = mask.nonzero()
        list_pred_label_ids = [indices[indices[:, 0] == i][:, 1].tolist() for i in range(pred_probs.size(0))]

        pred_labels = [[pred_id for pred_id in pred_label_ids] for pred_label_ids in list_pred_label_ids]
        pred_probs=pred_probs.tolist()
        return pred_labels, pred_probs

if __name__=="__main__":
    with open("/Users/guoxing.lan/projects/datasets/text_classification/reuters21578_cleaned/dict_id_to_label.json",'r',encoding="UTF-8") as rf:
        id_to_label=json.load(rf)
    list_text=[]
    list_true_labels=[]
    with open(input_data_path,'r',encoding="UTF-8") as rf:
        for line in rf:
            temp_dict=json.loads(line.strip())
            list_text.append(temp_dict["text"])
            list_true_labels.append(temp_dict["label_ids"])
    my_model=MyModel(model_name_or_path=model_name_or_path,
                     tokenizer_name_or_path=model_name_or_path,
                     device=device)
    pred_labels=[]
    pred_probs=[]
    list_text=list_text[0:16]
    list_true_labels=list_true_labels[0:16]
    for i in range(0,len(list_text), batch_size):
        start_index=i
        end_index=start_index+batch_size
        temp_list_text=list_text[start_index:end_index]
        temp_pred_labels, temp_pred_probs=my_model.batch_inference(temp_list_text)
        pred_labels+=temp_pred_labels
        pred_probs+=temp_pred_probs
    performance = evaluate(pred_labels, list_true_labels, id_to_label, threshold=0.5, top_k=None)
    performance_filename = "performance.json"
    with open(os.path.join(model_name_or_path,performance_filename), 'w', encoding="UTF-8") as wf:
        json.dump(performance, wf, ensure_ascii=False)
    print(json.dumps(performance, ensure_ascii=False))


    print(f'finish')











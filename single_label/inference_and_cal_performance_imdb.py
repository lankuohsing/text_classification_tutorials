from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
model_name_or_path="/Users/guoxing.lan/projects/github/text_classification_tutorials/outputs/imdb_bsz-2_lr-2e-05_len-256/checkpoint-16"


input_data_path="/Users/guoxing.lan/projects/github/text_classification_tutorials/jsonls/imdb_train_dev_test/test.jsonl"

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
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs_batch)
        pred_probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_label_ids = torch.argmax(pred_probs, dim=1).tolist()
        pred_labels = [int(self.model.config.id2label[pred_id]) for pred_id in pred_label_ids]
        pred_probs=pred_probs.tolist()
        return pred_labels, pred_probs

if __name__=="__main__":

    list_text=[]
    list_true_labels=[]
    with open(input_data_path,'r',encoding="UTF-8") as rf:
        for line in rf:
            temp_dict=json.loads(line.strip())
            list_text.append(temp_dict["text"])
            list_true_labels.append(temp_dict["label"])
    my_model=MyModel(model_name_or_path=model_name_or_path,
                     tokenizer_name_or_path=model_name_or_path,
                     device=device)
    pred_labels=[]
    pred_probs=[]
    for i in range(0,len(list_text), batch_size):
        start_index=i
        end_index=start_index+batch_size
        temp_list_text=list_text[start_index:end_index]
        temp_pred_labels, temp_pred_probs=my_model.batch_inference(temp_list_text)
        pred_labels+=temp_pred_labels
        pred_probs+=temp_pred_probs









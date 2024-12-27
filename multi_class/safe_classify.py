分析代码是否有问题
import os
import json
import time
import numpy as np
from datasets import load_dataset, Dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback, \
    DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import argparse


def load_tokenizer_and_model(tokenizer_path, model_path, num_labels):
    if "qwen2" in model_path.lower():
        kwargs = {"device_map": "auto"}
    else:
        kwargs = {}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=num_labels, **kwargs)
    if model.config.pad_token_id is None:
        print('设置pad_token_id')
        model.config.pad_token_id = tokenizer.eos_token_id
        print(f'pad_token_id: {model.config.pad_token_id}')
    return tokenizer, model


def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 将数据转换为Hugging Face的Dataset格式
    dataset = Dataset.from_list(data)
    return dataset


# 将字符串标签转换为整数标签
def encode_labels(example):
    example['label'] = label_feature.str2int(example['label'])
    return example


# 将字符串标签转换为整数标签
def encode_labels_binary(example):
    example['label'] = 0 if example['label'] == '正常' else 1
    return example


# 定义tokenizer函数
def tokenize_function(examples):
    inputs = tokenizer(examples['text'], max_length=128, truncation=True)
    inputs["labels"] = examples["label"]
    return inputs


# 定义评估指标
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    # precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
    #     p.label_ids, preds, average='micro')
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        p.label_ids, preds, average='macro')
    acc = accuracy_score(p.label_ids, preds)

    # 计算每个类别的F1分数
    precision_per_macro, recall_per_macro, f1_per_class, _ = precision_recall_fscore_support(
        p.label_ids, preds, average=None, labels=range(num_labels))

    # 将每个类别的F1分数添加到结果字典中
    metrics = {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro
    }

    for i, f1 in enumerate(f1_per_class):
        label = label_feature.int2str(i)
        metrics[f'f1_class_{label}'] = f1

    for i, precision in enumerate(precision_per_macro):
        label = label_feature.int2str(i)
        metrics[f'precision_class_{label}'] = precision

    for i, recall in enumerate(recall_per_macro):
        label = label_feature.int2str(i)
        metrics[f'recall_class_{label}'] = recall

    return metrics


# 定义评估指标
def compute_metrics_custom(preds, labels):
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro')
    acc = accuracy_score(labels, preds)

    # 计算每个类别的F1分数
    precision_per_macro, recall_per_macro, f1_per_class, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=range(num_labels))

    # 将每个类别的F1分数添加到结果字典中
    metrics = {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro
    }

    for i, f1 in enumerate(f1_per_class):
        label = label_feature.int2str(i)
        metrics[f'f1_class_{label}'] = f1

    for i, precision in enumerate(precision_per_macro):
        label = label_feature.int2str(i)
        metrics[f'precision_class_{label}'] = precision

    for i, recall in enumerate(recall_per_macro):
        label = label_feature.int2str(i)
        metrics[f'recall_class_{label}'] = recall

    return metrics


# 定义二分类评估指标
def compute_metrics_binary(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# 定义二分类评估指标
def compute_metrics_binary_custom(preds, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# 自定义回调函数来记录每个epoch的结果
class LogResultsCallback(TrainerCallback):
    def __init__(self, lr, bs):
        self.lr = lr
        self.bs = bs
        self.epoch_results = []
        self.best_result = None
        self.best_params = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            self.epoch_results.append(metrics)
            if self.best_result is None or metrics['eval_accuracy'] > self.best_result['eval_accuracy']:
                self.best_result = metrics
                self.best_params = {
                    'learning_rate': self.lr,
                    'batch_size': self.bs
                }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train',
                        help='train, val or test')
    parser.add_argument('--labels', type=str,
                        default='multi', help='binary or multi')
    parser.add_argument('--model_path', type=str,
                        default='/mnt/neimeng/nlp/projects/pretrain/xxx/model_hub/chinese-roberta-wwm-ext',
                        help='model path')
    parser.add_argument('--tokenizer_path', type=str,
                        default='/mnt/neimeng/nlp/projects/pretrain/xxx/model_hub/chinese-roberta-wwm-ext',
                        help='tokenizer path')
    parser.add_argument('--train_path', type=str,
                        default='/mnt/neimeng/nlp/projects/pretrain/xxx/program/mi_backend/shumei_backend/safe_classifier/data/train/title_train_0807.json',
                        help='train data path')
    parser.add_argument('--val_path', type=str,
                        default='/mnt/neimeng/nlp/projects/pretrain/xxx/program/mi_backend/shumei_backend/safe_classifier/data/test/title_test_0807.json',
                        help='val data path')
    parser.add_argument('--test_path', type=str,
                        default='/mnt/neimeng/nlp/projects/pretrain/xxx/program/mi_backend/shumei_backend/safe_classifier/data/test',
                        help='test data path')
    args = parser.parse_args()

    # 获取所有的标签
    if args.labels == 'binary':
        # all_labels = ['正常', '不正常']
        num_labels = 2
        encode_labels_fn = encode_labels_binary
        compute_metrics_fn = compute_metrics_binary
    else:
        all_labels = ["正常", "引战辱骂+攻击辱骂", "垃圾广告+垃圾广告", "危险行为引导+引人不适", "色情低俗+色情低俗", "富文本+富文本", "未成年不良+未成年",
                      "危险行为引导+自残自杀倾向",
                      "侵权+假冒官方", "危险行为引导+其它危险行为", "公序良俗+不文明行为", "涉政+国家名称", "公序良俗+劣迹艺人", "涉政+领导人", "引战辱骂+引战挂人",
                      "暴恐血腥+暴恐", "色情低俗+色情宣传", "涉政+其他"]
        # 创建ClassLabel特性
        label_feature = ClassLabel(names=list(all_labels))
        num_labels = len(label_feature.names)
        encode_labels_fn = encode_labels
        compute_metrics_fn = compute_metrics
    print(f"num_labels: {num_labels}")

    model_path = args.model_path
    model_name = os.path.basename(model_path)
    tokenizer_path = args.tokenizer_path
    # 加载模型和tokenizer
    tokenizer, model = load_tokenizer_and_model(
        tokenizer_path, model_path, num_labels)
    data_collator = DataCollatorWithPadding(tokenizer)

    if args.mode == 'train':
        train_path = args.train_path
        val_path = args.val_path
        train_name = os.path.basename(train_path).split('.')[0]
        val_name = os.path.basename(val_path).split('.')[0]
        train_data = load_dataset(train_path)
        val_data = load_dataset(val_path)

        train_dataset = train_data.map(encode_labels_fn)
        val_dataset = val_data.map(encode_labels_fn)

        # 对数据集进行tokenization
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)

        # 设置格式
        train_dataset.set_format(type='torch', columns=[
            'input_ids', 'attention_mask', 'label'])
        val_dataset.set_format(type='torch', columns=[
            'input_ids', 'attention_mask', 'label'])

        epoch_results = []
        epoch_nums = 10
        # Define ranges for the parameters you want to search
        learning_rates = [2e-5]
        batch_sizes = [64]

        # Perform a grid search over all combinations of parameters
        for lr in learning_rates:
            for bs in batch_sizes:
                all_results = []
                t = time.strftime('%Y%m%d%H%M%S', time.localtime())
                save_dir = f'./results/{train_name}_{model_name}_lr{lr}_bs{bs}_t{t}'
                os.makedirs(save_dir, exist_ok=True)
                if args.labels == 'multi':
                    # 保存标签映射
                    label_mapping = {i: label for i,
                                                  label in enumerate(label_feature.names)}
                    with open(os.path.join(save_dir, 'label_mapping.json'), 'w', encoding='utf-8') as f:
                        json.dump(label_mapping, f,
                                  ensure_ascii=False, indent=4)
                # Define the training arguments
                training_args = TrainingArguments(
                    output_dir=save_dir,
                    num_train_epochs=epoch_nums,
                    learning_rate=lr,
                    per_device_train_batch_size=bs,
                    per_device_eval_batch_size=bs,
                    warmup_steps=500,
                    weight_decay=0.01,
                    logging_steps=10,
                    eval_strategy='epoch',
                    save_strategy='epoch',
                    metric_for_best_model="accuracy",  # 选择最好的模型的指标
                )

                # Instantiate the callback
                log_results_callback = LogResultsCallback(lr=lr, bs=bs)

                # Instantiate trainer with updated args
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    compute_metrics=compute_metrics_fn,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    data_collator=data_collator,
                    callbacks=[log_results_callback]
                )
                # Train model
                trainer.train()

                # # Evaluate model
                # eval_result = trainer.evaluate()

                # Save all results and the best result to a single file
                results_to_save = {
                    'best_result': log_results_callback.best_result,
                    'best_params': log_results_callback.best_params,
                    'all_results': log_results_callback.epoch_results
                }
                with open(os.path.join(save_dir, 'training_results.json'), 'w', encoding='utf-8') as f:
                    json.dump(results_to_save, f, ensure_ascii=False, indent=4)
    elif args.mode == 'val':
        val_path = args.val_path
        val_name = os.path.basename(val_path).split('.')[0]
        val_data = load_dataset(val_path)
        val_dataset = val_data.map(encode_labels_fn)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        val_dataset.set_format(type='torch', columns=[
            'input_ids', 'attention_mask', 'label'])

        t = time.strftime('%Y%m%d%H%M%S', time.localtime())
        save_dir = os.path.join(model_path, val_name + f'_{t}')
        os.makedirs(save_dir, exist_ok=True)
        eval_bs = 32

        # 对测试数据进行预测
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=save_dir,
                per_device_eval_batch_size=eval_bs,
            ),
            data_collator=data_collator,
        )
        predictions = trainer.predict(val_dataset)
        # 计算评估指标
        metrics = compute_metrics_fn(predictions)
        # 保存评估指标
        with open(os.path.join(save_dir, 'test_metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)

        test_results = []
        preds = np.argmax(predictions.predictions, axis=1)
        for i, pred in enumerate(preds):
            if args.labels == 'binary':
                predicted_label = '正常' if pred == 0 else '不正常'
            else:
                predicted_label = label_feature.int2str(pred.item())
            test_results.append({
                'text': val_data[i]['text'],
                'label': val_data[i]['label'],
                'predicted_label': predicted_label
            })

        # 保存预测结果
        with open(os.path.join(save_dir, 'test_predictions.json'), 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=4)
    else:
        test_dir = args.test_path
        test_bs = 32
        for test_path in os.listdir(test_dir):
            if not test_path.endswith("title_test_0807.json"):
                continue
            test_path = os.path.join(test_dir, test_path)
            test_name = os.path.basename(test_path).split('.')[0]
            test_data = load_dataset(test_path)
            test_dataset = test_data.map(encode_labels_fn)
            test_dataset = test_dataset.map(tokenize_function, batched=True)
            test_dataset.set_format(type='torch', columns=[
                'input_ids', 'attention_mask', 'label'])

            t = time.strftime('%Y%m%d%H%M%S', time.localtime())
            save_dir = os.path.join(model_path, test_name + "_" + t)
            os.makedirs(save_dir, exist_ok=True)

            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=save_dir,
                    per_device_eval_batch_size=test_bs,
                ),
                data_collator=data_collator
            )
            p = trainer.predict(test_dataset)

            # 计算概率
            probs = np.exp(p.predictions) / np.sum(np.exp(p.predictions), axis=1, keepdims=True)

            thresholds = [0.9]

            for threshold in thresholds:
                preds = []
                for i, prob in enumerate(probs):
                    pred = np.argmax(prob)
                    if pred == 0:
                        if prob[0] <= threshold:
                            pred = np.argmax(prob[1:]) + 1
                    preds.append(pred)
                labels = p.label_ids

                # 计算多分类评估指标
                metrics = compute_metrics_custom(preds, labels)
                # 保存评估指标
                with open(os.path.join(save_dir, f'test_metrics_threshold_{threshold}.json'), 'w',
                          encoding='utf-8') as f:
                    json.dump(metrics, f, ensure_ascii=False, indent=4)
                test_results = []
                # 保存预测结果
                for i, pred in enumerate(preds):
                    if args.labels == 'binary':
                        predicted_label = '正常' if pred == 0 else '不正常'
                    else:
                        predicted_label = label_feature.int2str(int(pred))
                    test_results.append({
                        'text': test_data[i]['text'],
                        'label': test_data[i]['label'],
                        'predicted_label': predicted_label
                    })
                with open(os.path.join(save_dir, f'test_predictions_threshold_{threshold}.json'), 'w',
                          encoding='utf-8') as f:
                    json.dump(test_results, f, ensure_ascii=False, indent=4)

                # # 计算二分类评估指标
                # preds = [1 if pred != 0 else 0 for pred in preds]
                # labels = [1 if label != 0 else 0 for label in labels]
                # metrics = compute_metrics_binary_custom(preds, labels)
                # # 保存评估指标
                # with open(os.path.join(save_dir, f'test_metrics_binary_threshold_{threshold}.json'), 'w', encoding='utf-8') as f:
                #     json.dump(metrics, f, ensure_ascii=False, indent=4)

                # test_results = []
                # # 保存预测结果
                # for i, pred in enumerate(preds):
                #     predicted_label = '不正常' if pred == 1 else '正常'
                #     test_results.append({
                #         'text': test_data[i]['text'],
                #         'label': test_data[i]['label'],
                #         'predicted_label': predicted_label
                #     })
                # with open(os.path.join(save_dir, f'test_predictions_binary_threshold_{threshold}.json'), 'w', encoding='utf-8') as f:
                #     json.dump(test_results, f, ensure_ascii=False, indent=4)
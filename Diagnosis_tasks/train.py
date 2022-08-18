import logging
import torch
import argparse
import numpy as np
#from dataclass import load_dataset
from models import SeperateTaskForClassification
from datasets import load_dataset, load_metric
from datacollator import CustomCollatorWithPadding
from transformers import DataCollatorWithPadding, DataCollatorForTokenClassification


from transformers import  AutoTokenizer, AutoConfig, set_seed
from transformers import TrainingArguments, Trainer
logger = logging.getLogger(__name__)https://github.com/tjunlp-lab/TGEA/blob/main/Diagnosis_tasks/train.py
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-chinese")
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--dev_path", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--number_of_gpu", type=int, help="Number of available GPUs.")

    parser.add_argument("--batch_size_per_gpu", type=int, default=32,help='batch size for each gpu.')
    parser.add_argument("--test_batch_size_per_gpu", type=int, default=32, help='test batch size for each gpu.')
    parser.add_argument("--gradient_accumulation_steps", type=int, help="gradient accumulation step.")
    parser.add_argument("--effective_batch_size", type=int,
                        help='effective_bsz = batch_size_per_gpu x number_of_gpu x gradient_accumulation_steps')
    parser.add_argument("--total_steps", type=int,
                        help='total effective training steps for pre-training stage')
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--num_labels", type=int)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--do_train", type=bool, default=False)
    parser.add_argument("--do_eval", type=bool, default=False)
    parser.add_argument("--do_test", type=bool, default=False)
    parser.add_argument("--use_focal", type=bool, default=False)
    parser.add_argument("--focal_alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42, help='training seed')




    return parser.parse_args()



if __name__ == '__main__':
    cuda_available =  torch.cuda.is_available()
    if cuda_available:
        print("Cuda is available.")
    if cuda_available:
        if torch.cuda.device_count() > 1:
            multi_gpu_training = True
            print("Using Multi-GPU training, number of GPU is {}".format(torch.cuda.device_count()))
        else:
            print("Using single GPU training.")
    else:
        pass
    args = parse_config()
    device = torch.device('cuda')
    set_seed(args.seed)

    batch_size_per_gpu, gradient_accumulation_steps, number_of_gpu = \
    args.batch_size_per_gpu, args.gradient_accumulation_steps, args.number_of_gpu
    #effective_batch_size = batch_size_per_gpu * gradient_accumulation_steps * number_of_gpu

    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_files = {'train': args.train_path, 'validation': args.dev_path, 'test': args.test_path}
    # dataset = load_dataset(args.task_name, args.train_path, args.dev_path, args.test_path)
    dataset = load_dataset('json', data_files=data_files)
    pre_metrics = None
    metric_precision = load_metric("precision")
    metric_recall = load_metric("recall")
    metric_f1 = load_metric("f1")
    metric_accuracy = load_metric("accuracy")

    if args.task_name == 'ErroneousDetection':
        input_type = 'pooled_cls'
        classification_type = 'sequence'
        num_labels = 2

        def compute_metrics(eval_pred):
            results = {}
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            f1 = metric_f1.compute(predictions=predictions, references=labels)
            accuracy = metric_accuracy.compute(predictions=predictions, references=labels)
            precision = metric_precision.compute(predictions=predictions, references=labels)
            recall = metric_recall.compute(predictions=predictions, references=labels)
            results.update(accuracy)
            results.update(precision)
            results.update(recall)
            results.update(f1)
            return results

        def process_label(example):
            example['label'] = 0 if example['label'] == '正确' else 1
            return example
        def tokenizer_function(example):
            return tokenizer(example['text'])
        dataset = dataset.map(process_label)
        tokenized_datasets = dataset.map(tokenizer_function, batched=True)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        model = SeperateTaskForClassification(model_name,
                     input_type,
                     classification_type,
                     num_labels,
                     use_focal=args.use_focal,
                     focal_alpha=args.focal_alpha)
        metrics = compute_metrics


    elif args.task_name == 'MiSEWDetection':


        input_type = None
        classification_type = 'token'
        num_labels = 2

        def process_function(example):
            tokenized_text = tokenizer(example['text'].strip().lstrip().split(' '), is_split_into_words=True, return_offsets_mapping=True)
            raw_label = example.pop('label')
            label_ids = []
            word_ids = tokenized_text.word_ids()
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    label_ids.append(raw_label[word_idx])
            tokenized_text['labels'] = label_ids
            return tokenized_text
        tokenized_datasets = dataset.map(process_function)
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        def compute_metrics_mapping(trainer, dataset):
            predictions, labels, _ = trainer.predict(dataset)
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            offsets = dataset['offset_mapping']
            total_f1 = []
            total_accuracy = []
            total_precision = []
            total_recall = []
            text = dataset['text']
            nums = 0
            for i in range(len(offsets)):

                true_prediction = [p for (p,l) in zip(predictions[i], labels[i]) if l != -100]
                true_label= [l for (p,l) in zip(predictions[i], labels[i]) if l != -100]
                offset = [o for o in offsets[i] if (o[1] - o[0]) != 0]
                mapping_prediction = []
                mapping_label = []
                for j in range(len(true_label)):
                    mapping_prediction.extend([true_prediction[j]] * (offset[j][1] - offset[j][0]))
                    mapping_label.extend([true_label[j]] * (offset[j][1] - offset[j][0]))

                if  len(''.join(text[i].split(' '))) != len(mapping_prediction):
                    nums += 1
                f1 = metric_f1.compute(predictions=mapping_prediction, references=mapping_label)
                accuracy = metric_accuracy.compute(predictions=mapping_prediction, references=mapping_label)
                precision = metric_precision.compute(predictions=mapping_prediction, references=mapping_label)
                recall = metric_recall.compute(predictions=mapping_prediction, references=mapping_label)

                total_f1.append(f1['f1'])
                total_accuracy.append(accuracy['accuracy'])
                total_precision.append(precision['precision'])
                total_recall.append(recall['recall'])
            result = {
                      'accuracy': np.mean(total_accuracy),
                      'precision': np.mean(total_precision),
                      'recall':np.mean(total_recall),
                      'f1': np.mean(total_f1),
                      }
            return result

        model = SeperateTaskForClassification(model_name,
                     input_type,
                     classification_type,
                     num_labels,
                     use_focal=args.use_focal,
                     focal_alpha=args.focal_alpha)
        metrics = None

    elif 'spandetection' in args.task_name:
        input_type = None
        classification_type = 'token'
        num_labels = 3

        def process_function(example):
            tokenized_text = tokenizer(example['text'].strip().lstrip().split(' '), is_split_into_words=True,
                                       return_offsets_mapping=True)
            raw_label = example.pop('label')
            label_ids = []
            misew_ids = []
            word_ids = tokenized_text.word_ids()
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                    misew_ids.append(-100)
                else:
                    # 2 errorneous span
                    # 1 misew span
                    # 0 other span
                    if raw_label[word_idx] == 2:
                        label_ids.append(1)
                        misew_ids.append(1)
                    elif raw_label[word_idx] == 1:
                        label_ids.append(0)
                        misew_ids.append(0)
                    else:
                        label_ids.append(-100)
                        misew_ids.append(-100)
            tokenized_text['labels'] = label_ids
            tokenized_text['misew_labels'] = misew_ids
            return tokenized_text


        tokenized_datasets = dataset.map(process_function)
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


        def compute_metrics_mapping(trainer, dataset):
            predictions, labels, _ = trainer.predict(dataset)
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            offsets = dataset['offset_mapping']
            misew_labels = dataset['misew_labels']
            total_f1 = []
            total_accuracy = []
            total_precision = []
            total_recall = []
            total_f1_misew = []
            total_accuracy_misew = []
            total_precision_misew = []
            total_recall_misew = []
            text = dataset['text']
            for i in range(len(offsets)):
                true_prediction = [p for (p, ml) in zip(predictions[i], misew_labels[i]) if ml != -100]
                true_label = [l for (l, ml) in zip(labels[i], misew_labels[i]) if ml != -100]

                offset = [o for o in offsets[i] if (o[1] - o[0]) != 0]
                mapping_prediction = []
                mapping_label = []
                for j in range(len(true_label)):
                    mapping_prediction.extend([true_prediction[j]] * (offset[j][1] - offset[j][0]))
                    mapping_label.extend([true_label[j]] * (offset[j][1] - offset[j][0]))
                f1 = metric_f1.compute(predictions=mapping_prediction, references=mapping_label)
                accuracy = metric_accuracy.compute(predictions=mapping_prediction, references=mapping_label)
                precision = metric_precision.compute(predictions=mapping_prediction, references=mapping_label)
                recall = metric_recall.compute(predictions=mapping_prediction, references=mapping_label)
                '''
                true_prediction_misew = [p for (p, ml) in zip(predictions[i], misew_labels[i]) if ml != -100]
                true_label_misew = [l for (l, ml) in zip(labels[i], misew_labels[i]) if ml != -100]

                for idx in range(len(true_prediction_misew)):
                    if true_prediction_misew[idx] == 2:
                        true_prediction_misew[idx] = 0
                    else:
                        true_prediction_misew[idx] = 1
                    if true_label_misew[idx] == 2:
                        true_label_misew[idx] = 0
                    else:
                        true_label_misew[idx] = 1

                #offset = [o for o in offsets[i] if (o[1] - o[0]) != 0]
                mapping_prediction_misew = []
                mapping_label_misew = []
                for j in range(len(true_label_misew)):
                    mapping_prediction_misew.extend([true_prediction_misew[j]] * (offset[j][1] - offset[j][0]))
                    mapping_label_misew.extend([true_label_misew[j]] * (offset[j][1] - offset[j][0]))
                f1_misew = metric_f1.compute(predictions=mapping_prediction_misew, references=mapping_label_misew)
                accuracy_misew = metric_accuracy.compute(predictions=mapping_prediction_misew, references=mapping_label_misew)
                precision_misew = metric_precision.compute(predictions=mapping_prediction_misew, references=mapping_label_misew)
                recall_misew = metric_recall.compute(predictions=mapping_prediction_misew, references=mapping_label_misew)
                '''

                total_f1.append(f1['f1'])
                total_accuracy.append(accuracy['accuracy'])
                total_precision.append(precision['precision'])
                total_recall.append(recall['recall'])
                '''
                total_f1_misew.append(f1_misew['f1'])
                total_accuracy_misew.append(accuracy_misew['accuracy'])
                total_precision_misew.append(precision_misew['precision'])
                total_recall_misew.append(recall_misew['recall'])
                '''
            result = {
                      'accuracy': np.mean(total_accuracy),
                      'precision': np.mean(total_precision),
                      'recall':np.mean(total_recall),
                      'f1': np.mean(total_f1),
                        #
                        # 'misew_accuracy': np.mean(total_accuracy_misew),
                        # 'misew_precision': np.mean(total_precision_misew),
                        # 'misew_recall': np.mean(total_recall_misew),
                        # 'misew_f1': np.mean(total_f1_misew),

                      }
            return result

        model = SeperateTaskForClassification(model_name,
                                              input_type,
                                              classification_type,
                                              num_labels,
                                              use_focal=args.use_focal,
                                              focal_alpha=args.focal_alpha)
        metrics = None#compute_metrics

    elif 'ErroneousClassification' in args.task_name:
        #input_type = 'span'
        input_type = 'pooled_cls'
        classification_type = 'sequence'
        if args.task_name[-1] == '1':
            label_maps = {'搭配错误':0,
                          '残缺错误':1,
                          '成分多余':2,
                          '语篇错误':3,
                          '常识错误':4}

        num_labels = len(label_maps)

        def process_function(example):

            tokenized_text = tokenizer(example['text'].strip().lstrip().split(' '), is_split_into_words=True, return_offsets_mapping=True)
            span_id = example['span']
            error_span_mask = []
            for i in tokenized_text.word_ids():
                if i == span_id:
                    error_span_mask.append(1)
                else:
                    error_span_mask.append(0)

            tokenized_text['error_span_mask'] = error_span_mask
            if args.task_name[-1] == '1':
                tokenized_text['labels'] = label_maps[example['class1']]
            elif args.task_name[-1] == '2':
                tokenized_text['labels'] = label_maps[example['class2']]
            return tokenized_text
        tokenized_datasets = dataset.map(process_function)

        data_collator = CustomCollatorWithPadding(tokenizer=tokenizer)

        def compute_metrics(eval_pred):
            results = {}
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            accuracy = metric_accuracy.compute(predictions=predictions, references=labels)
            f1 = metric_f1.compute(predictions=predictions, references=labels, average="macro")
            precision =  metric_precision.compute(predictions=predictions, references=labels, average="macro")
            recall =  metric_recall.compute(predictions=predictions, references=labels, average="macro")
            results.update(accuracy)
            results.update(precision)
            results.update(recall)
            results.update(f1)
            return results
        model = SeperateTaskForClassification(model_name,
                     input_type,
                     classification_type,
                     num_labels,
                     use_focal=args.use_focal,
                     focal_alpha=args.focal_alpha,
                     use_erroneous_span=True)
        metrics = compute_metrics

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size_per_gpu,
        per_device_eval_batch_size=args.test_batch_size_per_gpu,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_steps=100000,
        save_total_limit=1,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if args.do_eval else None,
        compute_metrics=metrics,
        preprocess_logits_for_metrics=pre_metrics if args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
    if args.do_eval:
        logger.info("*** Evaluate ***")
        if args.task_name == 'MiSEWDetection' or args.task_name == 'spandetection':
            metrics = compute_metrics_mapping(trainer, tokenized_datasets['validation'])
        else:
            metrics = trainer.evaluate()
        #metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
    if args.do_test:
        logger.info("*** Evaluate ***")

        if args.task_name == 'MiSEWDetection' or args.task_name == 'spandetection':
            metrics = compute_metrics_mapping(trainer, tokenized_datasets['test'])
        else:
            metrics = trainer.evaluate(tokenized_datasets["test"])
        trainer.log_metrics("test", metrics)





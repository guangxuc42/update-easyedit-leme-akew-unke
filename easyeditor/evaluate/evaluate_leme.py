import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate
from tqdm.auto import tqdm
import nltk
from datasets import Dataset
from tqdm.auto import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from transformers import (
    AutoTokenizer,
    DebertaV2ForSequenceClassification,
    AutoModelForCausalLM,
    pipeline
)
# from src.utils import get_sample_id
import hashlib
import json
import argparse
import os

# construct sample id from sample
# use md5 hash of requested rewrite
def get_sample_id(sample):
    if 'requested_rewrite' in sample:
        return hashlib.md5(
            json.dumps(sample["requested_rewrite"]).encode()
        ).hexdigest()
    else:
        return hashlib.md5(
            json.dumps(sample).encode()
        ).hexdigest()


def generate_sample_token(model,model_name_hparams,tok,prompt,device):
    batch = tokenizer(
        prompt,
        return_tensors='pt',
        padding=True
    )

    sampling_params = {}
    sampling_params = {
        'do_sample': True,
        'top_k': 50,
        'top_p': 0.95,
        'temperature': 0.9,
        'num_return_sequences': 1
    }
    
    generated_sample = sample
    with torch.no_grad():
        post_edit_outputs = edited_model.generate(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            max_new_tokens=600,
            repetition_penalty=1.1,
            **sampling_params
        )
    generate_sample=tok.decode(
            post_edit_outputs.detach().cpu().numpy().tolist(), skip_special_tokens=True).replace(prompt, '').strip()

    return generate_sample

def compute_automatic_metrics(samples):
    if args.metric == 'nli':
        logger.info('Getting NLI scores')
        nli_model = DebertaV2ForSequenceClassification.from_pretrained(
            "Joelzhang/deberta-v3-large-snli_mnli_fever_anli_R1_R2_R3-nli",
            local_files_only=True
        )
        nli_tokenizer = AutoTokenizer.from_pretrained(
            "Joelzhang/deberta-v3-large-snli_mnli_fever_anli_R1_R2_R3-nli",
            local_files_only=True
        )
        nli_pipe = pipeline(
            "text-classification", 
            model=nli_model,
            tokenizer=nli_tokenizer,
            device=0 if device == 'cuda' else -1
        )
        results = get_nli_scores(
            samples,
            nli_pipe
        )
    elif args.metric == 'perplexity':
        logger.info('Getting Perplexity scores')
        perplexity_tokenizer = AutoTokenizer.from_pretrained(
            'gpt2-xl',
            local_files_only=True
        )
        perplexity_model = AutoModelForCausalLM.from_pretrained(
            'gpt2-xl',
            local_files_only=True
        )
        perplexity_model = perplexity_model.to(device)
        results = get_perplexity_scores(
            samples,
            perplexity_model,
            perplexity_tokenizer
        )
    elif args.metric == 'rouge':
        logger.info('Getting ngram overlap scores')
        results = get_ngram_overlap_scores(samples)


def get_overlap_measures(sample):
    subject = sample['requested_rewrite']['subject']
    related_entity = sample['coupled_prompts_and_properties']['coupled_entities'][0]['entity']
    new_fact = sample["requested_rewrite"]["prompt"].format(
            sample["requested_rewrite"]['subject']
        ) + sample["requested_rewrite"]['target_new']['str']
    old_fact = sample["requested_rewrite"]["prompt"].format(
            sample["requested_rewrite"]['subject']
        ) + sample["requested_rewrite"]['target_true']['str']
    passage_of_text_about_subject_of_edit = sample['subject_prompt'].strip().replace('\n', ' ')
    passage_of_text_about_related_entity = sample['coupled_prompt'].strip().replace('\n', ' ')

    return {
        "subject_and_main_passage": {
            "predictions": [subject],
            "references": [passage_of_text_about_subject_of_edit]
        },
        "related_entity_and_main_passage": {
            "predictions": [related_entity],
            "references": [passage_of_text_about_subject_of_edit]
        },
        "subject_and_related_passage": {
            "predictions": [subject],
            "references": [passage_of_text_about_related_entity]
        },
        "related_entity_and_related_passage": {
            "predictions": [related_entity],
            "references": [passage_of_text_about_related_entity]
        },
        "old_fact_and_main_passage": {
            "predictions": [old_fact],
            "references": [passage_of_text_about_subject_of_edit]
        },
        "old_fact_and_related_passage": {
            "predictions": [old_fact],
            "references": [passage_of_text_about_related_entity]
        },
        "new_fact_and_main_passage": {
            "predictions": [new_fact],
            "references": [passage_of_text_about_subject_of_edit]
        },
        "new_fact_and_related_passage": {
            "predictions": [new_fact],
            "references": [passage_of_text_about_related_entity]
        }
    }


def get_ngram_overlap_scores(samples):
    all_results = {}
    for sample in tqdm(samples):
        results = {}
        overlap_measures = get_overlap_measures(sample)
        for key, value in overlap_measures.items():
            if key not in results:
                results[key] = []
            results[key].append(
                rouge.compute(
                    predictions=value['predictions'],
                    references=value['references']
                )
            )
        all_results[get_sample_id(sample)] = results
    return all_results


def calculate_perplexity(
    passage,
    model,
    tokenizer
):
    inputs = tokenizer(passage, return_tensors='pt', truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
    loss = outputs.loss
    perplexity = torch.exp(loss)
    return perplexity.item()


def get_perplexity_scores(
    samples,
    model,
    tokenizer
):
    all_results = {}
    for sample in tqdm(samples):
        perplexity_scores = {
            "main_passage": [],
            "related_passage": []
        }
        passage_of_text_about_subject_of_edit = sample['subject_prompt'].strip().replace('\n', ' ')
        passage_of_text_about_related_entity = sample['coupled_prompt'].strip().replace('\n', ' ')
        perplexity_scores['main_passage'].append(
            calculate_perplexity(
                passage_of_text_about_subject_of_edit,
                model,
                tokenizer
            )
        )
        perplexity_scores['related_passage'].append(
            calculate_perplexity(
                passage_of_text_about_related_entity,
                model,
                tokenizer
            )
        )
        all_results[get_sample_id(sample)] = perplexity_scores
    return all_results


def construct_nli_dataset(sample):
    subject_ground_truth = sample['coupled_prompts_and_properties']['subject_entity']['ground_truth']
    subject_ground_truth_string = '- ' + '\n- '.join([f"{key}: {', '.join(value)}" for key,value in subject_ground_truth.items()])
    related_entity_ground_truth = sample['coupled_prompts_and_properties']['coupled_entities'][0]['ground_truth']
    related_entity_ground_truth_string = '- ' + '\n- '.join([f"{key}: {', '.join(value)}" for key,value in related_entity_ground_truth.items()])

    new_fact = sample["requested_rewrite"]["prompt"].format(
            sample["requested_rewrite"]['subject']
        ) + " " + sample["requested_rewrite"]['target_new']['str']
    old_fact = sample["requested_rewrite"]["prompt"].format(
            sample["requested_rewrite"]['subject']
        ) + " " +  sample["requested_rewrite"]['target_true']['str']
    passage_of_text_about_subject_of_edit = sample['subject_prompt'].strip().replace('\n', ' ')
    passage_of_text_about_related_entity = sample['coupled_prompt'].strip().replace('\n', ' ')
    main_text_segmented = nltk.tokenize.sent_tokenize(passage_of_text_about_subject_of_edit)
    related_text_segmented = nltk.tokenize.sent_tokenize(passage_of_text_about_related_entity)

    return {
        "new_fact_and_main_passage": [
            f"{new_fact}. {sent}"
            for sent in main_text_segmented
        ],
        "old_fact_and_main_passage": [
            f"{old_fact}. {sent}"
            for sent in main_text_segmented
        ],
        "new_fact_and_related_passage": [
            f"{new_fact}. {sent}"
            for sent in related_text_segmented
        ],
        "old_fact_and_related_passage": [
            f"{old_fact}. {sent}"
            for sent in related_text_segmented
        ],
        "ground_truth_and_main_passage": [
            f"{subject_ground_truth_string}. {sent}"
            for sent in main_text_segmented
        ],
        "ground_truth_and_related_passage": [
            f"{related_entity_ground_truth_string}. {sent}"
            for sent in related_text_segmented
        ],
        "main_passage_consistency": [
            f"{sent_1}. {sent_2}"
            for sent_1 in main_text_segmented
            for sent_2 in main_text_segmented
            if sent_1 != sent_2
        ],
        "related_passage_consistency": [
            f"{sent_1}. {sent_2}"
            for sent_1 in related_text_segmented
            for sent_2 in related_text_segmented
            if sent_1 != sent_2
        ],
        # TODO(dom): should make this a list of lists for doc matrix
        "main_passage_and_related_passage": [
            f"{sent_1}. {sent_2}"
            for sent_1 in main_text_segmented
            for sent_2 in related_text_segmented
        ]
    }


def get_dataset(sample, dkey):
    dataset = {
        dkey: []
    }
    constructed_dataset = construct_nli_dataset(sample)
    for key, value in constructed_dataset.items():
        if key == dkey:
            dataset[key].extend(value)
    return Dataset.from_dict(dataset)


def get_nli_scores(samples, nli_pipe):
    dataset_keys = [
        "new_fact_and_main_passage",
        "old_fact_and_main_passage",
        "new_fact_and_related_passage",
        "old_fact_and_related_passage",
        "ground_truth_and_main_passage",
        "ground_truth_and_related_passage",
        "main_passage_and_related_passage",
        "main_passage_consistency",
        "related_passage_consistency"
    ]

    results = {}
    for sample in samples:
        sample_results = {}
        for dkey in tqdm(dataset_keys):
            ds = get_dataset(sample, dkey)

            nli_results = []
            for out in tqdm(nli_pipe(KeyDataset(ds, dkey), batch_size=8, truncation="only_first", top_k=None), total=len(ds)):
                nli_results.append(
                    out
                )

            key_results = {}
            for nli_res in nli_results:
                for result in nli_res:
                    if result['label'] not in key_results:
                        key_results[result['label']] = []
                    key_results[result['label']].append(result['score'])

            sample_results[dkey] = key_results

        results[get_sample_id(sample)] = sample_results
    return results
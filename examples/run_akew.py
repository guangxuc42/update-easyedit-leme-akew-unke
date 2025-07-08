import os.path
import sys
import json
import argparse

sys.path.append('..')
from easyeditor import (
    FTHyperParams,
    MEMITHyperParams,
    ROMEHyperParams,
    UnKEHyperParams,
    BaseEditor,
    summary_metrics,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--edit_type', required=True, type=str,
                        choices=['struct', 'unstruct', 'extract'])
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--data_type', required=True, type=str,
                        choices=['CounterFact', 'MQuAKE-CF', 'WikiUpdate'])
    parser.add_argument('--output_dir', default='./outputs', type=str)
    parser.add_argument('--ds_size', default=3, type=int)
    parser.add_argument('--sequential_edit', action="store_true")

    args = parser.parse_args()

    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'UnKE':
        editing_hparams = UnKEHyperParams
    else:
        raise NotImplementedError

    K = args.ds_size

    if args.data_type == 'CounterFact':
        edit_data = json.load(open(f'{args.data_dir}/{args.data_type}.json', 'r', encoding='utf-8'))[:K]

        prompts = [edit_data_['requested_rewrite']['prompt_full'] for edit_data_ in edit_data]
        subject = [edit_data_['requested_rewrite']['subject'] for edit_data_ in edit_data]
        rephrase_prompts = [edit_data_['paraphrase_prompts'][0] for edit_data_ in edit_data]
        target_new = [edit_data_['requested_rewrite']['answer_new'] for edit_data_ in edit_data]
        uns_target_new = [edit_data_['requested_rewrite']['fact_new_uns'] for edit_data_ in edit_data]
        
        extract_prompts = [
            edit_data_['requested_rewrite']['unsfact_triplets_GPT'][0]['prompt'].format(edit_data_['requested_rewrite']['unsfact_triplets_GPT'][0]['subject']) 
            for edit_data_ in edit_data
        ]

        extract_subject = [
            edit_data_['requested_rewrite']['unsfact_triplets_GPT'][0]['subject']
            for edit_data_ in edit_data
        ]

        extract_target_new = [
            edit_data_['requested_rewrite']['unsfact_triplets_GPT'][0]['target'] 
            for edit_data_ in edit_data
        ]

    elif args.data_type == 'MQuAKE-CF':
        edit_data = json.load(open(f'{args.data_dir}/{args.data_type}.json', 'r', encoding='utf-8'))[:K]

        prompts = [edit_data_['requested_rewrite'][0]['prompt'].format(edit_data_['requested_rewrite'][0]['subject']) for edit_data_ in edit_data]
        subject = [edit_data_['requested_rewrite'][0]['subject'] for edit_data_ in edit_data]
        target_new = [edit_data_['requested_rewrite'][0]['target_new']['str'] for edit_data_ in edit_data]
        uns_target_new = [edit_data_['requested_rewrite'][0]['fact_new_uns'] for edit_data_ in edit_data]
        
        extract_prompts = [
            edit_data_['requested_rewrite']['unsfact_triplets_GPT'][0]['prompt'].format(edit_data_['requested_rewrite']['unsfact_triplets_GPT'][0]['subject']) 
            for edit_data_ in edit_data
        ]

        extract_subject = [
            edit_data_['requested_rewrite']['unsfact_triplets_GPT'][0]['subject']
            for edit_data_ in edit_data
        ]

        extract_target_new = [
            edit_data_['requested_rewrite']['unsfact_triplets_GPT'][0]['target'] 
            for edit_data_ in edit_data
        ]

        
    elif args.data_type == 'WikiUpdate':
        edit_data = json.load(open(f'{args.data_dir}/{args.data_type}.json', 'r', encoding='utf-8'))[:K]

        prompts = [edit_data_['requested_rewrite']['prompt'].format(edit_data_['requested_rewrite']['subject']) for edit_data_ in edit_data]
        subject = [edit_data_['requested_rewrite']['subject'] for edit_data_ in edit_data]
        target_new = [edit_data_['requested_rewrite']['target_new']['str'] for edit_data_ in edit_data]
        uns_target_new = [edit_data_['requested_rewrite']['fact_new_uns'] for edit_data_ in edit_data]
        
        extract_prompts = [
            edit_data_['requested_rewrite']['unsfact_triplets_GPT'][0]['prompt'].format(edit_data_['requested_rewrite']['unsfact_triplets_GPT'][0]['subject']) 
            for edit_data_ in edit_data
        ]

        extract_subject = [
            edit_data_['requested_rewrite']['unsfact_triplets_GPT'][0]['subject']
            for edit_data_ in edit_data
        ]

        extract_target_new = [
            edit_data_['requested_rewrite']['unsfact_triplets_GPT'][0]['target'] 
            for edit_data_ in edit_data
        ]

    if args.edit_type == 'struct':
        prompts = prompts
        target_new = target_new
    elif args.edit_type == 'unstruct':
        prompts = prompts
        target_new = uns_target_new
    elif args.edit_type == 'extract':
        prompts = extract_prompts
        target_new = extract_target_new
        subject = extract_subject

    hparams = editing_hparams.from_hparams(f'{args.hparams_dir}')

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f'{hparams.model_name.split("/")[-1]}_{args.editing_method}_N={args.ds_size}_Sequential={args.sequential_edit}.json'
        )

    print("See results at: ", output_file)

    eval_metric = {
        'CounterFact': 'token em',
        'MQuAKE-CF': 'token em',
        'WikiUpdate': 'token em'
    }

    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts if args.data_type == 'CounterFact' else None,
        target_new=target_new,
        subject=subject,
        locality_inputs=None,
        portability_inputs=None,
        sequential_edit=args.sequential_edit,
        eval_metric=eval_metric[args.data_type]
    )

    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    if len(metrics) > 0:
        summary_metrics(metrics)


import os.path
import sys
import json
import argparse

sys.path.append('..')
from easyeditor import (
    FTHyperParams,
    MEMITHyperParams,
    ROMEHyperParams,
    BaseEditor,
    summary_metrics,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--data_type', required=True, type=str,
                        choices=['CounterFact','ZsRE'])
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
    else:
        raise NotImplementedError

    K = args.ds_size

    if args.data_type == 'CounterFact':
        edit_data = json.load(open(f'{args.data_dir}/counterfact_with_coupled_entities.json', 'r', encoding='utf-8'))[:K]

        prompts = [edit_data_['requested_rewrite']['prompt'].format(edit_data_['requested_rewrite']['subject']) for edit_data_ in edit_data]
        subject = [edit_data_['requested_rewrite']['subject'] for edit_data_ in edit_data]
        rephrase_prompts = [edit_data_['paraphrase_prompts'][0] for edit_data_ in edit_data]
        target_neg = [edit_data_['requested_rewrite']['target_true']['str'] for edit_data_ in edit_data]
        target_new = [edit_data_['requested_rewrite']['target_new']['str'] for edit_data_ in edit_data]

        subject_prompts = [edit_data_['coupled_prompts_and_properties']['subject_entity']['coupled_prompt'] + \
        f"\n-relationship to {edit_data_['coupled_prompts_and_properties']['coupled_entities'][0]['entity']}" + f"\n\n{edit_data_['requested_rewrite']['subject']}"
        for edit_data_ in edit_data]

        related_entities = [edit_data_['coupled_prompts_and_properties']['coupled_entities'][0]['entity'] for edit_data_ in edit_data]
        #TODO overlap和non-overlap
        related_entity_ground_truth = [edit_data_['coupled_prompts_and_properties']['coupled_entities'][0]['overlapping_ground_truth'] for edit_data_ in edit_data]
        subject_ground_truth = [edit_data_['coupled_prompts_and_properties']['subject_entity']['ground_truth'] for edit_data_ in edit_data]

        coupled_prompts = [edit_data_['coupled_prompts_and_properties']['coupled_entities'][0]['coupled_prompt'] + \
            f"\n-relationship to {edit_data_['requested_rewrite']['subject']}" + f"\n\n{edit_data_['coupled_prompts_and_properties']['coupled_entities'][0]['entity']}"
        for edit_data_ in edit_data
        ]
        
        leme_inputs = {
            'subject_prompts': subject_prompts,
            'coupled_prompts': coupled_prompts,
            'related_entities': related_entities,
            'related_entity_ground_truth': related_entity_ground_truth,
            'subject_ground_truth': subject_ground_truth
        }

    elif args.data_type == 'ZsRE':
        edit_data = json.load(open(f'{args.data_dir}/zsre_mend_eval_with_coupled_entities.json', 'r', encoding='utf-8'))[:K]

        prompts = [edit_data_['src'] for edit_data_ in edit_data]
        subject = [edit_data_['subject'] for edit_data_ in edit_data]
        rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in edit_data]
        target_neg = [edit_data_['answers'][0] for edit_data_ in edit_data]
        target_new = [edit_data_['alt'] for edit_data_ in edit_data]
        locality_prompts = [edit_data_['loc'] for edit_data_ in edit_data]
        locality_ans = [edit_data_['loc_ans'] for edit_data_ in edit_data]
        locality_inputs = {
            'neighborhood':{
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }

        subject_prompts = [edit_data_['coupled_prompts_and_properties']['subject_entity']['coupled_prompt'] + \
        f"\n-relationship to {edit_data_['coupled_prompts_and_properties']['coupled_entities'][0]['entity']}" + f"\n\n{edit_data_['subject']}"
        for edit_data_ in edit_data]

        related_entities = [edit_data_['coupled_prompts_and_properties']['coupled_entities'][0]['entity'] for edit_data_ in edit_data]
        #TODO overlap和non-overlap
        related_entity_ground_truth = [edit_data_['coupled_prompts_and_properties']['coupled_entities'][0]['overlapping_ground_truth'] for edit_data_ in edit_data]
        subject_ground_truth = [edit_data_['coupled_prompts_and_properties']['subject_entity']['ground_truth'] for edit_data_ in edit_data]

        coupled_prompts = [edit_data_['coupled_prompts_and_properties']['coupled_entities'][0]['coupled_prompt'] + \
            f"\n-relationship to {edit_data_['subject']}" + f"\n\n{edit_data_['coupled_prompts_and_properties']['coupled_entities'][0]['entity']}"
        for edit_data_ in edit_data
        ]
        
        leme_inputs = {
            'subject_prompts': subject_prompts,
            'coupled_prompts': coupled_prompts,
            'related_entities': related_entities,
            'related_entity_ground_truth': related_entity_ground_truth,
            'subject_ground_truth': subject_ground_truth
        }

        

    hparams = editing_hparams.from_hparams(f'{args.hparams_dir}')

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f'{hparams.model_name.split("/")[-1]}_{args.editing_method}_N={args.ds_size}_Sequential={args.sequential_edit}.json'
        )

    print("See results at: ", output_file)

    eval_metric = {
        'CounterFact': 'token em',
        'ZsRE': 'token em'
    }

    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        target_neg=target_neg,
        subject=subject,
        locality_inputs=locality_inputs if args.data_type == 'ZsRE' else None,
        portability_inputs=None,
        leme_inputs=leme_inputs,
        sequential_edit=args.sequential_edit,
        eval_metric=eval_metric[args.data_type]
    )

    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    if len(metrics) > 0:
        summary_metrics(metrics)


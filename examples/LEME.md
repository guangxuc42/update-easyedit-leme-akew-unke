# Long-form evaluation of model editing

This README is about reproducing the paper [Long-form evaluation of model editing](https://arxiv.org/abs/2402.09394).

## Table of Contents

- [Dataset Description](#Dataset-Description)
- [Running Examples of Using AKEW](#Running-Examples-of-Using-AKEW)
---

## Dataset Description

**LEME** introduces a novel benchmark to evaluate the effectiveness of model editing in long-form generation scenarios, addressing limitations of existing short-form metrics. It proposes a new evaluation protocol based on paragraph-level outputs, incorporating both human-annotated and automatic measures such as edit consistency, factual consistency, internal consistency, topicality, and naturalness.
**Note:** You can download their dataset from [Github](https://github.com/domenicrosati/longform-evaluation-model-editing/tree/main/data).

The data format of **LEME** are as follows:

There are three datasets: **CounterFact** and **ZsRE** in JSON.  
A sample in them looks like

    {
        "case_id": 3,
        "pararel_idx": 6791,
        "requested_rewrite": {
            "prompt": "{}, which is located in",
            "relation_id": "P17",
            "target_new": {
                "str": "Sweden",
                "id": "Q34"
            },
            "target_true": {
                "str": "Spain",
                "id": "Q29"
            },
            "subject": "Autonomous University of Madrid"
        },
        "paraphrase_prompts": [
            "and Sallie Beavers Riley. Autonomous University of Madrid is located in",
            "Houston, Tex: Anson Jones Press. Autonomous University of Madrid, located in"
        ],
        "neighborhood_prompts": [
            "Biure is located in",
            ...
        ],
        ...
        "id": "ed0a981c3bb1ac994eb621a0590739a9",
        "coupled_prompts_and_properties": {
            "subject_entity": {
                "properties": [
                    "language used",
                    "has subsidiary",
                    "owner of",
                    "member of",
                    "has part(s)",
                    "office held by head of the organization",
                    "rector",
                    "country",
                    "located in the administrative territorial entity"
                ],
                "coupled_prompt": "Write an essay about Autonomous University of Madrid\nInclude the following information:\n- language used\n- has subsidiary\n- owner of\n- member of\n- has part(s)\n- office held by head of the organization\n- rector\n- country\n- located in the administrative territorial entity",
                "ground_truth": {
                    "country": [
                        "Spain"
                    ],
                    "located in the administrative territorial entity": [
                        "Madrid"
                    ],
                    "rector": [
                        "Pedro Sanz Mart\u00ednez"
                    ],
                    "member of": [
                        "Alliance 4 Universities",
                        ...
                    ],
                    "has subsidiary": [
                        "Instituto de Investigaciones Biom\u00e9dicas Alberto Sols",
                        ...
                    ],
                    "office held by head of the organization": [
                        "Rector of the Autonomous University of Madrid"
                    ],
                    "owner of": [
                        "Instituto de F\u00edsica Te\u00f3rica UAM/CSIC",
                        ...
                    ],
                    "has part(s)": [
                        "Biology Department"
                    ],
                    "language used": [
                        "Spanish"
                    ]
                },
                "entity": "Autonomous University of Madrid",
                "entity_id": "Q788091"
            },
            "coupled_entities": [
                {
                    "entity": "Instituto de Investigaciones Biom\u00e9dicas Alberto Sols",
                    "coupled_prompt": "Write an essay about Instituto de Investigaciones Biom\u00e9dicas Alberto Sols\n    Include the following information:\n- parent organization\n- owned by\n- country\n- located in the administrative territorial entity",
                    "mutual_properties": [],
                    "subject_as_object": [
                        "parent organization",
                        "owned by"
                    ],
                    "target_true_as_object": [],
                    "overlap_properties": [
                        "country",
                        "located in the administrative territorial entity"
                    ],
                    "original_property_of_subject_as_object": [],
                    "overlapping_ground_truth": {
                        "country": [
                            "Spain"
                        ],
                        "located in the administrative territorial entity": [
                            "Madrid"
                        ],
                        "owned by": [
                            "Spanish National Research Council",
                            "Autonomous University of Madrid"
                        ],
                        "parent organization": [
                            "Autonomous University of Madrid",
                            "Spanish National Research Council"
                        ]
                    },
                    ...
                    "additional_properties": [
                        "named after",
                        "field of work"
                    ],
                    "entity_id": "Q5918253"
                },
                ...
            ],
            "coupled_properties_count": 22
        }
    }

The explanations are
- 'case_id': Unique identifier for this editing case
- 'pararel_idx': Index linking to the Pararel dataset (relation template repository)
- `requested_rewrite`: The requested rewrite information for editing, with the following fields:
    - `prompt`: The prompt template.
    - `relation_id`: Wikidata property ID (e.g., "P17" = country).
    - `target_new`: Desired edit output.
        - `str`: Entity name.
        - `id`: Wikidata QID.
    - `target_true`: Original factual state.
        - `str`: Entity name.
        - `id`: Wikidata QID.
    - `subject`: The subject.
- `paraphrase_prompts`: Variant phrasings testing edit generalization
- `neighborhood_prompts`: Unrelated queries testing locality preservation
- `id`: Universal unique identifier for this sample
- `coupled_prompts_and_properties`: LEME-specific long-form evaluation framework
  - `subject_entity`: Primary edited entity details
    - `properties`: Attributes to include in generated text
    - `coupled_prompt`: Instruction for paragraph generation
    - `ground_truth`: Pre-edit factual attributes (key-value pairs)
    - `entity`: Formal name of subject
    - `entity_id`: Wikidata QID
  - `coupled_entities`: Related entities for consistency checks
    - `entity`: Name of related entity
    - `coupled_prompt`: Generation instructions for related entity
    - `mutual_properties`: Shared attributes with subject
    - `subject_as_object`: Relations where subject is the object
    - `overlap_properties`: Geographically/topically shared properties
    - `overlapping_ground_truth`: Factual attributes of related entity
    - `additional_properties`: Entity-specific attributes
    - `entity_id`: Wikidata QID
  - `coupled_properties_count`: Total properties across entities

## Running Examples of LEME

If you want to know how to easily use EasyEdit with the **LEME** dataset. You can follow the steps below:

1️⃣ Create a new directory `EasyEdit/data/LEME` and download the files such as `counterfact_with_coupled_entities.json` from [GitHub](https://github.com/domenicrosati/longform-evaluation-model-editing/tree/main/data) into this folder.

2️⃣ Run the main experiment with:
```bash
bash run_leme.sh
```
The `run_leme.sh` script includes a sample command just like:
```
python run_leme.py \
 --editing_method=FT \
 --hparams_dir=../hparams/FT/qwen2.5-7b.yaml \
 --data_dir=../data/LEME \
 --ds_size=2 \
 --data_type=ZsRE \
```
The result output is a lot. we figure out what it's meaning in the [paper](https://arxiv.org/abs/2402.09394):
` Edit consistency Subject`:`subject_and_main_passage`
` Edit consistency Related`:`subject_and_related_passage`
` Factual consistency Subject`:`related_entity_and_main_passage`
` Factual consistency Related`:`related_entity_and_related_passage`
` Internal consistency Subject`:`ground_truth_and_main_passage`
` Internal consistency Related`:`ground_truth_and_related_passage`
` Internal consistency Cross`:`main_passage_consistency`
` Topicality`: `related_passage_consistency`
` Naturalness`: `main_passage_and_related_passage`

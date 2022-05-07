#CUP
This repository is for CUP: Curriculum Learning based Prompt Tuning for Implicit Event Argument Extraction


## Dataset
Download RAMS and WikiEvent datasets into ./data folder.

## Checkpoints
Our checkpoints are available [here](https://drive.google.com/drive/folders/1IDAuOWxIlStmgzkmgFsd24Ckx9tJ3QhR).

Download them into ./experiments
## Evaluation 
Evaluate performance on RAMS

    script/test_RAMS.sh

Evaluate performance on WikiEvents

    sh script/test_Wikievents.sh


## Reproduce the results by yourself
In training stage, we utilize documental AMR graph. Hence, preprocess data first.
### Transition-based AMR decoder
Follow the instructions in this [repository](https://github.com/IBM/transition-amr-parser) to train an AMR parser.

Then use ./data/amr_parser.py to parse the sentences of the two datasets.

### Coreference Resolution
We utilize ready-made coreference resolution tool available [here](https://demo.allennlp.org/coreference-resolution)
### Use your own AMR decoder or coreference resolutioner
Process training data into the same form as data/WikiEvents/amrs/train.amr.txt and data/WikiEvents/corefered.json

### Combine coref with raw training data to build documental AMR graph

For RAMS: 

    python ./data/preprocess.py --train_dir=./data/RAMS/train.jsonlines --coref_dir=./data/RAMS/corefered.json --output_dir=./data/RAMS/RAMSwithcore/train.jsonl

For WikiEvents: 

    python DataProcessers.py # First transform WikiEvent data into RAMS form  
    python ./data/preprocess.py --train_dir=./data/wikievents/informative/train.jsonl --coref_dir=./data/wikievents/corefered.json --output_dir=./data/wikievents/WikiwCoref/informative/train.jsonl

### Training
For full data training, store all doc_keys into f'./{args.data_path}/doc_keys.jsonl' 

To conduct few-shot training, first select a particular ratio of samples, then store their doc_keys into f'./{args.data_path}/doc_keys.jsonl' 

Train on RAMS:

    sh scripts/train_RAMS.sh
Train on WikiEvents:

    sh scripts/train_WikiEvents.sh


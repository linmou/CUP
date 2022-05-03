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


## Reimplement by yourself
In training stage, we utilize documental AMR graph. Hence, preprocess data first.
### Transition-based AMR decoder

### Coreference Resolution

### use your own AMR decoder or coreference resolutioner
Process training data into the same form as data/WikiEvents/amrs/train.amr.txt and data/WikiEvents/corefered.json

### Training
To train on RAMS:

    sh scripts/train_RAMS.sh
to train on WikiEvents:

    sh scripts/train_WikiEvents.sh


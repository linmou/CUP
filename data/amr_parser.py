from transition_amr_parser.parse import AMRParser
import os
import ipdb
import json
import re
import spacy
spacier=spacy.load('en_core_web_sm')
parser = AMRParser.from_checkpoint('transition-amr-parser-action-pointer/DATA/AMR2.0/models/exp_cofill_o8.3_act-states_RoBERTa-large-top24/_act-pos-grh_vmask1_shiftpos1_ptr-lay6-h1_grh-lay123-h2-allprev_1in1out_cam-layall-h2-abuf/ep120-seed42/checkpoint_wiki.smatch_top3-avg.pt')
all_amrs = []
err=0
with open('/amr_based_prompt/data/wikievents/WikiwCoref/informative/train.jsonl') as f:
    all_dicts = f.readlines()
    Last_err = False
    for line in all_dicts:
        instance = json.loads(line)
        # if instance['event_key']['id'] == 'wiki_ied_bombings_5_news_4-E18':
        #     Last_err = True
        # if Last_err == False: continue
        doc=[]
        for sent in instance['sentences']:
            sent_tk=[]
            for tk in sent:
                tk = re.sub(r'[\u201c\u201d]', '\"', tk)
                tk = re.sub(r'[\u2019\u2018]', '\'', tk)
                tk = re.sub(r'[—–]', '-', tk)
                sent_tk.append(tk)
            doc.append(sent_tk)

        all_amrs.append('evt_mention:'+instance["doc_key"]+'\n') # TODO:sentence id

        i=0
        while i <len(doc):
            try:
                annotations = parser.parse_sentences(doc[i:i+10])
            except:
                # ipdb.set_trace()
                annotations[0][0]='WrongSample'+annotations[0][0]
                err += 1
            i += 10

            # no matter right or wrong, writedown amrs to take the space , so that the index of this file can be
            # consistent with downstream tasks
            all_amrs.extend(annotations[0])

print(err)
with open('/amr_based_prompt/data/wikievents/amr/train.amr.txt', 'w', encoding='utf8') as f:
    f.writelines(all_amrs)
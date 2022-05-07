from copy import deepcopy
import random
import ipdb
from openprompt.data_utils.utils import InputExample
import os
import json, csv
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable,Optional
import re
import spacy
spacier = spacy.load('en_core_web_sm')
def set_custom_boundaries(doc):
    '''Sometimes, spacy set < and > as end of sentence.
    This custom boundary will fix that bug.  '''
    for token in doc[:-1]:
        if ">" in token.text or "<" in token.text :
            doc[token.i].is_sent_start = False
            doc[token.i+1].is_sent_start = False
    return doc
#add custom boundary once
spacier.add_pipe(set_custom_boundaries, before="parser")


from openprompt.utils.logging import logger
from openprompt.data_utils.data_processor import DataProcessor

from DocumentalAMR import DocumentalAmrGraph,load_amrs_to_graph,parse_RAMS,load_document_amr

class RAMSProcesser (DataProcessor):

    def __init__(self):
        super().__init__()
        self.labels = None

    def get_event_type(self,ex):
        evt_type = []
        for evt in ex['evt_triggers']:
            for t in evt[2]:
                evt_type.append( t[0])

        return evt_type

    def load_ontology(self, ontology_file):
        # read ontology
        ontology_dict = {}
        with open(ontology_file,'r',encoding='utf8') as f:
            for lidx, line in enumerate(f):
                if lidx == 0:  # header
                    continue
                fields = line.strip().split(',')
                if len(fields) < 2:
                    break
                evt_type = fields[0]
                args = fields[2:]

                ontology_dict[evt_type] = {
                    'template': fields[1]
                }

                for i, arg in enumerate(args):
                    if arg != '':
                        ontology_dict[evt_type]['arg{}'.format(i + 1)] = arg
                        ontology_dict[evt_type][arg] = 'arg{}'.format(i + 1)

        return ontology_dict

    def create_gold_gen(self, ex, ontology_dict, mark_trigger=True):
        '''assumes that each line only contains 1 event.
        Input: <s> Template with special <arg> placeholders </s> </s> Passage </s>
        Output: <s> Template with arguments and <arg> when no argument is found.
        '''

        evt_type = self.get_event_type(ex)[0]
        context_words = [w for sent in ex['sentences'] for w in sent]
        template = ontology_dict[evt_type.replace('n/a', 'unspecified')]['template'].replace('.',',')
        input_template = re.sub(r'<arg\d>', '<arg>', template)
        for triple in ex['gold_evt_links']:
            trigger_span, argument_span, arg_name = triple
            try:
                arg_num = ontology_dict[evt_type.replace('n/a', 'unspecified')][arg_name]
            except:
                continue # err annotation
            arg_text = ' '.join(context_words[argument_span[0]:argument_span[1] + 1])
            template = re.sub('<{}>'.format(arg_num), arg_text, template)
        template = re.sub(r'<arg\d>', '<arg>', template) # to guarantee no arg\d remains in templates
        # replace <arg> with arg_type
        # for empty_arg_num in re.findall(r'<arg\d>', template):
        #     arg_name=ontology_dict[evt_type.replace('n/a', 'unspecified')][empty_arg_num[1:-1]]
        #     arg_name=re.split(r'\d+',arg_name)[-1]
        #     template = template.replace(f'{empty_arg_num}', arg_name)
            # arg_endpos=template.find(arg_name) + len(arg_name)+1
            # if arg_endpos<=len(template)-1 and arg_name.startswith(template[arg_endpos:arg_endpos+min(len(empty_arg_num),3)]):
            #     template=re.sub(empty_arg_num,'', template)
            # else:
            #     template=template.replace(f'{empty_arg_num}',arg_name)


        return input_template,template



    def get_examples(self, data_dir: str, split: str,**kwargs) -> List[InputExample]:
        examples,ex_dicts,amrs,= [],[],[]
        path = os.path.join(data_dir, "{}.jsonl".format(split))

        ontology_dict=self.load_ontology(kwargs['ontology_file'])
        doc_keys=json.load(open(os.path.join(data_dir,'doc_keys', "{}.json".format(split)))) # for few_shot learning
        with open(path) as f:
            all_dict = f.readlines()
            cnt=0
            for i, line in enumerate(all_dict):
                instance = json.loads(line)
                if instance['doc_key'] not in doc_keys or ('evt_key' in instance.keys() and instance['evt_key'] not in doc_keys):
                    continue
                event_type = self.get_event_type(instance)[0]
                trigger_id = instance["evt_triggers"][0][0]

                evt_links=deepcopy(instance['gold_evt_links'])
                sentences_tk = sum(instance['sentences'], [])
                try:
                    trigger_text=sentences_tk[trigger_id]
                except:
                    ipdb.set_trace()
                    trigger_text=instance['trigger_text']

                if kwargs['data_part']=='full':
                    input_template, tgt_text = self.create_gold_gen(instance, ontology_dict)

                else: # curriculum learnig

                    ex = deepcopy(instance)
                    if evt_links == []:
                        input_template, tgt_text = self.create_gold_gen(instance, ontology_dict)
                        if kwargs['data_part']=='easy':
                            sentences_tk=instance['sentences'][0]
                    else : # to find sent with trigger inside
                        sent_len = 0
                        sent_w_trg = 0
                        for sentid in range(len(instance['sentences'])):
                            if sent_len < trigger_id and (sent_len + len(instance['sentences'][sentid])) > trigger_id:
                                boundery = (sent_len, sent_len + len(instance['sentences'][sentid]))
                                sent_w_trg = sentid
                                break
                            sent_len += len(instance['sentences'][sentid])


                        if kwargs['data_part']=='easy':# intra sentence learning
                            try:
                                ex['sentences'] = [instance['sentences'][sent_w_trg]]
                            except:
                                ipdb.set_trace()
                            sentences_tk=instance['sentences'][sent_w_trg]

                            for ent in ex["ent_spans"]:
                                ent[0]-=boundery[0]
                                ent[1] -= boundery[0]
                            for trg in ex["evt_triggers"]:
                                trg[0]-=boundery[0]
                                trg[1] -= boundery[0]

                            for lkid,link in enumerate(evt_links): # modify links inside boundary
                                arg_spn=link[1]
                                if arg_spn[1]<boundery[1] and arg_spn[0]>=boundery[0]:
                                    ex['gold_evt_links'][lkid][1]=[arg_spn[0]-boundery[0],arg_spn[1]-boundery[0]]
                                    ex['gold_evt_links'][lkid][0]=[link[0][0]-boundery[0],link[0][1]-boundery[0]]

                            for lkid, link in enumerate(evt_links): # remove links outside boundary
                                arg_spn = link[1]
                                if arg_spn[1] < boundery[0] or arg_spn[0] >= boundery[1]:
                                    ex['gold_evt_links'].remove(link)

                            input_template, tgt_text = self.create_gold_gen(ex, ontology_dict)

                        elif kwargs['data_part']=='Given2For13': # sentence1,3 learning
                            ex['sentences'] = instance['sentences'][sent_w_trg - 1:sent_w_trg + 2]
                            newbdry0 = boundery[0] -  (len(instance['sentences'][sent_w_trg - 1]) if sent_w_trg-1 >=0 else 0)
                            newbdry1 = boundery[1] +  (len(instance['sentences'][sent_w_trg + 1]) if sent_w_trg + 1 < len(instance['sentences']) else 0)
                            boundery=(newbdry0,newbdry1)

                            inner_boundary = (len(sum(instance['sentences'][:sent_w_trg], [])),
                                              len(sum(instance['sentences'][:sent_w_trg + 1], [])))

                            arg_texts=[]
                            for lkid,link in enumerate(evt_links): # modify links inside boundary
                                arg_spn = link[1]
                                if arg_spn[1]<boundery[1] and arg_spn[0]>boundery[0]:
                                    ex['gold_evt_links'][lkid][1]=[arg_spn[0]-boundery[0],arg_spn[1]-boundery[0]]
                                    ex['gold_evt_links'][lkid][0]=[link[0][0]-boundery[0],link[0][1]-boundery[0]]

                                    if arg_spn[1]<inner_boundary[0] or arg_spn[0]>inner_boundary[1]:
                                        arg_texts.append(' '.join(sentences_tk[arg_spn[0]:arg_spn[1] + 1]))

                            for lkid, link in enumerate(evt_links): # remove links outside boundary
                                arg_spn = link[1]
                                if arg_spn[1] < boundery[0] or arg_spn[0] > boundery[1]:
                                    ex['gold_evt_links'].remove(link)

                            _, tgt_text = self.create_gold_gen(ex, ontology_dict)
                            input_template = tgt_text
                            for arg in arg_texts:
                                input_template = input_template.replace(arg, '<arg>')

                            sentences_tk = sum(ex['sentences'], [])

                        elif kwargs['data_part']=='Given123For04': # sentence1,3 learning
                            inner_boundary = (len(sum(instance['sentences'][:sent_w_trg-1], [])),
                                              len(sum(instance['sentences'][:sent_w_trg + 2], [])))

                            arg_texts=[]
                            # record=False
                            for lkid,link in enumerate(evt_links): # record spans outside inner boundery
                                arg_spn = link[1]
                                if arg_spn[1]<inner_boundary[0] or arg_spn[0] >= inner_boundary[1]:
                                    arg_texts.append(' '.join(sentences_tk[arg_spn[0]:arg_spn[1] + 1]))
                                    # record=True
                            # if len(evt_links)<3: record=False

                            _, tgt_text = self.create_gold_gen(ex, ontology_dict)
                            input_template = tgt_text
                            for arg in arg_texts:
                                input_template = input_template.replace(arg, '<arg>')

                        elif kwargs['data_part']=='Given2For0134': # inter sentence learning
                            inter_args=[]
                            for link in evt_links:# collect args outside boundary
                                arg_spn=link[1]
                                if arg_spn[1]<boundery[0] or arg_spn[0]>boundery[1]:
                                    inter_args.append(' '.join(sentences_tk[arg_spn[0]:arg_spn[1]]))

                            _, tgt_text = self.create_gold_gen(ex, ontology_dict)
                            input_template=tgt_text
                            for arg in inter_args:
                                input_template=input_template.replace(arg,'<arg>')

                doc = ' '.join(sentences_tk)
                doc = re.sub(r'[\u201c\u201d]', '\"', doc)
                doc = re.sub(r'[\u2019\u2018]', '\'', doc)
                doc = re.sub(r'[—–]', '-', doc)
                try:
                    doc = re.sub(trigger_text, '<trg>' + trigger_text + '<trg>', doc)
                except:
                    ipdb.set_trace()
                example = InputExample(guid=str(i) + '.' + split, text_a=doc,
                                       text_b=input_template, tgt_text=tgt_text, label=1,
                                       meta={'event_type': event_type, 'evt_links': evt_links,
                                             # 'sentences': instance['sentences'],
                                             'trigger_text': trigger_text, 'trigger_id': trigger_id,
                                             'doc_key': instance['doc_key']})
                # ex_dicts.append(ex)
                # if record:
                examples.append(example)

            if 'k_shot' in kwargs and kwargs['k_shot']!=None:
                random.shuffle(examples)
                examples=examples[:kwargs['k_shot']]

        print(cnt)
        return examples#,ex_dicts

    def get_src_tgt_len_ratio(self, ):
        pass

class WikiEventProcesser(DataProcessor):
    def __init__(self):
        super(WikiEventProcesser, self).__init__()
        self.MAX_CONTEXT_LENGTH = 300  # measured in words
        self.MAX_LENGTH = 512
        self.MAX_TGT_LENGTH = 70

        PRONOUN_FILE = 'pronoun_list.txt'
        self.pronoun_set = set()
        with open(PRONOUN_FILE, 'r') as f:
            for line in f:
                self.pronoun_set.add(line.strip())

    def check_pronoun(self,text):
        if text.lower() in self.pronoun_set:
            return True
        else:
            return False

    def clean_mention(self,text):
        '''
        Clean up a mention by removing 'a', 'an', 'the' prefixes.
        '''
        prefixes = ['the ', 'The ', 'an ', 'An ', 'a ', 'A ']
        for prefix in prefixes:
            if text.startswith(prefix):
                return text[len(prefix):]
        return text

    def load_ontology(self, ontology_file):
        '''
        Read ontology file for event to argument mapping.
        '''
        ontology_dict = {}
        with open(ontology_file, 'r') as f:
            ontology_dict = json.load(f)

        for evt_name, evt_dict in ontology_dict.items():
            for i, argname in enumerate(evt_dict['roles']):
                evt_dict['arg{}'.format(i + 1)] = argname
                # argname -> role is not a one-to-one mapping
                if argname in evt_dict:
                    evt_dict[argname].append('arg{}'.format(i + 1))
                else:
                    evt_dict[argname] = ['arg{}'.format(i + 1)]

        return ontology_dict

    def create_gold_gen(self, ex, ontology_dict, mark_trigger=True, index=0, ent2info=None, use_info=False,span_range=4,inner_boundary=0):
        '''
        If there are multiple events per example, use index parameter.

        Input: <s> Template with special <arg> 2 </s> </s> Passage </s>
        Output: <s> Template with arguments and <arg> when no argument is found.
        '''
        if use_info and ent2info == None:
            raise ValueError('entity to informative mention mapping required.')

        evt_type = ex['event_mentions'][index]['event_type']
        sentences=ex['sentences']
        sentID_w_trigger=ex['event_mentions'][index]['trigger']['sent_id']
        template = ontology_dict[evt_type]['template']
        # input_template = re.sub(r'<arg\d>', '<arg>', template)

        role2arg = defaultdict(list)

        for argument in ex['event_mentions'][index]['arguments']:
            role2arg[argument['role']].append(argument)

        role2arg = dict(role2arg)
        # create output template
        arg_idx2text = defaultdict(list)

        for role in role2arg.keys():
            if role not in ontology_dict[evt_type]:
                # annotation error
                continue
            for i, argument in enumerate(role2arg[role]):
                use_arg = True
                if use_info:
                    ent_id = argument['entity_id']
                    if ent_id in ent2info:
                        arg_text = self.clean_mention(ent2info[ent_id])
                        if self.check_pronoun(arg_text):
                            # skipping this argument
                            use_arg = False
                            # if arg_text != argument['text']:
                            # print('Original mention:{}, Informative mention:{}'.format(argument['text'], arg_text))
                    else:
                        arg_text = argument['text']
                else:
                    arg_text = argument['text']

                # assign the argument index
                if i < len(ontology_dict[evt_type][role]):
                    # enough slots to fill in
                    arg_idx = ontology_dict[evt_type][role][i]

                else:
                    # multiple participants for the same role
                    arg_idx = ontology_dict[evt_type][role][-1]

                if use_arg:
                    arg_idx2text[arg_idx].append(arg_text)


        ## select contents
        trigger = ex['event_mentions'][index]['trigger']
        offset = 0
        # trigger span does not include last index
        context_words = ex['tokens']
        center_sent = trigger['sent_idx']
        document=[]
        if len(context_words) > self.MAX_CONTEXT_LENGTH:
            cur_len = len(ex['sentences'][center_sent][0])
            context_words = [tup[0] for tup in ex['sentences'][center_sent][0]]
            if cur_len > self.MAX_CONTEXT_LENGTH:
                # one sentence is very long
                trigger_start = trigger['start']
                start_idx = max(0, trigger_start - self.MAX_CONTEXT_LENGTH // 2)
                # end_idx = min(len(context_words), trigger_start + self.MAX_CONTEXT_LENGTH // 2)
                context_words = context_words[start_idx: start_idx+self.MAX_CONTEXT_LENGTH]
                # context_words = context_words[start_idx: end_idx]
                offset = start_idx

            else:
                # take a sliding window
                left = center_sent - 1
                right = center_sent + 1

                total_sents = len(ex['sentences'])
                prev_len = 0
                while cur_len > prev_len:
                    prev_len = cur_len
                    # try expanding the sliding window
                    if left >= 0:
                        left_sent_tokens = [tup[0] for tup in ex['sentences'][left][0]]
                        if cur_len + len(left_sent_tokens) <= self.MAX_CONTEXT_LENGTH:
                            context_words = left_sent_tokens + context_words
                            left -= 1
                            cur_len += len(left_sent_tokens)

                    if right < total_sents:
                        right_sent_tokens = [tup[0] for tup in ex['sentences'][right][0]]
                        if cur_len + len(right_sent_tokens) <= self.MAX_CONTEXT_LENGTH:
                            context_words = context_words + right_sent_tokens
                            right += 1
                            cur_len += len(right_sent_tokens)
                # for sid in range(left+1,right):
                #     document.append([tup[0] for tup in ex['sentences'][sid][0]])

                # update trigger offset
                offset = sum([len(ex['sentences'][idx][0]) for idx in range(left + 1)])

        assert (len(context_words) <= self.MAX_CONTEXT_LENGTH)


        trigger['start'] = trigger['start'] - offset
        trigger['end'] = trigger['end'] - offset
        if mark_trigger:
            prefix = ' '.join(context_words[:trigger['start']])
            tgt = ' '.join(context_words[trigger['start']: trigger['end']])
            suffix = ' '.join(context_words[trigger['end']:])
            context = prefix + ' <trg>' + tgt + '<trg> '+ suffix
        else:
            context = ' '.join(context_words)

        doc=spacier(context)
        trg_pos=-1
        contextls=[]
        clues = ''
        for sid,sent in enumerate(doc.sents):
            if '<trg>' in sent.text:
                trg_pos=sid
            if '<trg>' not in sent.text and 'trg' in sent.text :
                ipdb.set_trace()
        for sid,sent in enumerate(doc.sents):
            if sid>=trg_pos-span_range and sid<=trg_pos+span_range:
                contextls.append(sent.text)
            if inner_boundary>=0 and sid >= trg_pos - inner_boundary and sid <= trg_pos + inner_boundary:
                clues+=(' '+sent.text)

        context=' '.join(contextls)
        if context== '':ipdb.set_trace()

        # measure the impact of context_len
        omitted_num=0
        arg_mentions=sum(list(arg_idx2text.values()),[])
        for arg in arg_mentions:
            if arg not in context:
                omitted_num+=1

        # develop clue prompts
        input_template=template
        for arg_idx, text_list in arg_idx2text.items():
            tmpls=text_list.copy()
            for argtxt in text_list:
                if argtxt not in clues:
                    tmpls.remove(argtxt)
            if tmpls==[]:continue
            text=' and '.join(tmpls)
            input_template = re.sub('<{}>'.format(arg_idx), text, input_template)

        # develop gold template
        output_template=template
        argtp2argtxt=dict()
        for arg_idx, text_list in arg_idx2text.items():
            text = ' and '.join(text_list)
            output_template = re.sub('<{}>'.format(arg_idx), text, output_template)
            arg_type=ontology_dict[evt_type][arg_idx]
            argtp2argtxt[arg_type] = text_list

        input_template = re.sub(r'<arg\d>', '<arg>', input_template)
        output_template = re.sub(r'<arg\d>', '<arg>', output_template)

        return input_template, output_template, contextls,context_words, omitted_num, argtp2argtxt

    def get_examples(self, data_dir: Optional[str] = None, split: Optional[str] = None,data_part:str='full',recordAsRAMS=False,**kwarg) -> List[InputExample]:
        ontology_dict = self.load_ontology(kwarg['ontology_file'])
        coref_split = 'dev' if split == 'val' else split
        coref_dir=os.path.join(data_dir,'coref')
        coref_reader = open(os.path.join(coref_dir, '{}.jsonlines'.format(coref_split)))
        examples=[]
        omitted_num,ttl_arg_num,ttl_doc_tks,skipped_ent=0,0,0,0
        doc_keys=json.load(open(os.path.join(data_dir,'doc_keys', "{}.json".format(split)))) # for few_shot learning
        with open(os.path.join(data_dir, '{}.jsonl'.format(split)), 'r') as reader:
            for step,(line, coref_line) in enumerate(zip(reader, coref_reader)):
                ex = json.loads(line.strip())
                corefs = json.loads(coref_line.strip())
                assert (ex['doc_id'] == corefs['doc_key'])
                if ex['doc_id'] not in doc_keys:
                    continue
                # mapping from entity id to information mention
                ent2info = {}
                for cidx, cluster in enumerate(corefs['clusters']):
                    for eid in cluster:
                        ent2info[eid] = corefs['informative_mentions'][cidx]

                for i in range(len(ex['event_mentions'])):
                    if split == 'train' and len(ex['event_mentions'][i]['arguments']) == 0:
                        continue
                    evt_type = ex['event_mentions'][i]['event_type']

                    if evt_type not in ontology_dict:  # should be a rare event type
                        continue

                    if data_part=='easy':
                        span_range=0
                        inner_boundary=-1 # no inner boundary
                    elif data_part=='Given2For13':
                        span_range=1
                        inner_boundary=0
                    elif data_part=='Given123For04' :
                        span_range=4
                        inner_boundary=1
                    elif data_part=='full':
                        span_range=4
                        inner_boundary=-1 # no inner boundary

                    input_template, output_template, contextls,context_words,omitted_args,argtp2argtxt = self.create_gold_gen(ex, ontology_dict,
                                                                                    index=i, ent2info=ent2info,
                                                                                    use_info=kwarg['use_info'],
                                                                                    span_range=span_range,inner_boundary=inner_boundary)
                    if recordAsRAMS:
                        trigger_span=[ex['event_mentions'][i]['trigger']['start'],ex['event_mentions'][i]['trigger']['end']-1] # -1 to fit the RAMS form, no need to -offset cause we hanled in create_gold_gen already
                        evt_triggers=[[trigger_span[0],trigger_span[1],[[ex['event_mentions'][i]['event_type']]]]]
                        entys={}
                        for enty in ex['entity_mentions']:
                            entys[enty['id']]=deepcopy(enty)

                        ent_spans=[]
                        gold_evt_links=[]
                        for arg in ex['event_mentions'][i]['arguments']:
                            start=entys[arg['entity_id']]['start']
                            end=entys[arg['entity_id']]['end']
                            ent_spans.append([start,end,[[arg['role'],1.0]]])
                            gold_evt_links.append([trigger_span,[start,end],arg['role']])

                        example = InputExample(guid=str(step) + '.' + str(i) + '.' + split, text_a=' '.join(contextls),
                                               label=1,
                                               tgt_text=output_template,
                                               text_b=input_template,
                                               meta={'doc_key':ex['event_mentions'][i]['id'],
                                                     'gold_evt_links':gold_evt_links,
                                                     'evt_triggers': evt_triggers, 'ent_spans': ent_spans,
                                                     'sentences': context_words,
                                                     }
                                               )

                    else:
                        trigger_text=ex['event_mentions'][i]['trigger']['text']
                        ttl_arg_num+=ex['event_mentions'][i]['arguments'].__len__()
                        ttl_doc_tks+=len((' '.join(contextls)).split(' '))
                        omitted_num+=omitted_args

                        example = InputExample(guid=str(step)+'.'+ str(i) + '.' + split, text_a=' '.join(contextls), label=1,
                                               tgt_text=output_template,
                                               text_b=input_template,
                                               meta={'evt_type':evt_type,'doc_key':ex['doc_id'],'evt_key':ex['event_mentions'][i]['id'],
                                                    'arguments':argtp2argtxt,'trigger_text':trigger_text,'sentences':contextls,
                                                    }
                                               )
                    examples.append(example)
                if (step+1)%100==0:
                    print(f'{step+1} documents handled.')

        print(f'{omitted_num} args are discarded due to the limitation of context len.')
        print(f'{ttl_arg_num} args in total. Avg doc_len ={ttl_doc_tks/len(examples)}')


        return examples

class AmrGraphProcesser(DataProcessor):

    def __init__(self):
        super().__init__()
        self.labels = None

    def print_path(self,path,sentences_tk):
        path_text=[]
        intersentence = False
        last_pos = 0

        for nid, node in enumerate(path):
            if node == {}:  continue
            # TODO: currently we give up parallel path and only choose one of the shortest paths, future work is to decode all paths
            for n, pathdir in node:
                if nid == 0:
                    begin = ' '.join(sentences_tk[n[0]:n[1]])
                if nid == len(path) - 1:
                    end = ' '.join(sentences_tk[n[0]:n[1]])
                try:
                    if n[1] == 'c' or n[1] == 's':
                        intersentence = True

                    else:
                        if pathdir == 1 or pathdir[0] != '-':
                            path_text.insert(last_pos, ' '.join(sentences_tk[n[0]:n[1]]))
                        elif pathdir[0] == '-':
                            last_pos += 1
                            path_text.insert(last_pos, ' '.join(sentences_tk[n[0]:n[1]]))
                except:
                    ipdb.set_trace()
                break  # we only decode one path



        return path_text,intersentence,begin,end


    def get_examples(self, data_dir: str, split: str, dataset='RAMS',**kwargs) -> List[InputExample]:
        assert 'train' in split , 'amr graphs are only contained in training data'
        examples, amrs = [], []
        path = os.path.join(data_dir, "{}.jsonl".format(split))
        if not os.path.exists(path):
            path = os.path.join(data_dir, "{}.jsonlines".format(split))

        with open(kwargs['amr_path'], encoding='utf8') as f:
            amrs = f.read()
        amrs = amrs.split('\n\n')
        amrs = amrs[:-1] if amrs[-1] == "" else amrs

        amr_id = 0
        doc_keys=json.load(open(os.path.join(data_dir,'doc_keys', "{}.json".format(split)))) # for few_shot learning
        with open(path) as f:
            all_dict = f.readlines()
            err,err_doc = 0,0
            for i, line in enumerate(all_dict):
                instance = json.loads(line)
                if instance['doc_key'] not in doc_keys:
                    continue
                if dataset=='RAMS':
                    event_type = instance["evt_triggers"][0][2][0][0]
                    trg_span=(instance['evt_triggers'][0][0],instance['evt_triggers'][0][1])
                    arg_span2type=dict()
                    for links in instance["gold_evt_links"]:
                        arg_span2type[tuple(links[1])]=re.split('[\d]+', links[2])[-1]
                    keys=[ sp[0] for sp in sorted(arg_span2type.items(), key=lambda x: abs((x[0][0] + x[0][1]) / 2 - trg_span[0]))] # sort by distance from trg to arg
                    values=[ sp[1] for sp in sorted(arg_span2type.items(), key=lambda x: abs((x[0][0] + x[0][1]) / 2 - trg_span[1]))]
                    arg_span2type=dict(zip(keys,values))

                    sentences_tk = sum(instance['sentences'], [])
                    doc=' '.join(sentences_tk)
                    doc = re.sub(r'[\u201c\u201d]', '\"', doc)
                    doc = re.sub(r'[\u2019\u2018]', '\'', doc)
                    doc = re.sub(r'[—–]', '-', doc)
                    sentences_tk = (doc).split(' ')
                    trigger_text = ' '.join(sentences_tk[trg_span[0]:trg_span[1] + 1])

                    documental_amrs = amrs[amr_id:amr_id + len(instance['sentences'])]
                    tokens = documental_amrs[0].split('\n')[0]
                    if not tokens.startswith('# ::tok '):
                        if tokens.startswith('WrongSample'):
                            amr_id += 1
                            while not amrs[amr_id].startswith('evt_mention'):
                                amr_id += 1
                            err_doc += 1
                            continue
                        else:
                            ipdb.set_trace()

                    tokens = tokens[len("# ::tok "):].split(' ')
                    if tokens[:min(5,len(tokens)-1)]!=sentences_tk[:min(5,len(tokens)-1)] or 'clusters' not in instance: # <ROOT> <trg> # or
                        if 'clusters'  in instance:
                            # print(tokens[:7],sentences_tk[:7])
                            amr_id += len(json.loads(all_dict[i - 1])['sentences'])  # due to the errs in amr parser are solved by copying previous doc's amr, so we + previous len(sentences)
                            # ipdb.set_trace()
                        else:
                            amr_id += len(instance['sentences'])
                        err_doc+=1
                        continue

                    amr_id += len(instance['sentences'])

                    graph = load_amrs_to_graph(documental_amrs)
                    graph.merge_coref({"clusters": instance['clusters'], 'document': instance['document']})
                    if not graph.tokens[0][0] == sentences_tk[0]: ipdb.set_trace()

                elif dataset=='wikievents':

                    event_type = instance["evt_triggers"][0][2][0][0]
                    trg_span=(instance['evt_triggers'][0][0],instance['evt_triggers'][0][1])
                    arg_span2type=dict()
                    for links in instance["gold_evt_links"]:
                        arg_span2type[tuple(links[1])]=re.split('[\d]+', links[2])[-1]
                    keys=[ sp[0] for sp in sorted(arg_span2type.items(), key=lambda x: abs((x[0][0] + x[0][1]) / 2 - trg_span[0]))] # sort by distance from trg to arg
                    values=[ sp[1] for sp in sorted(arg_span2type.items(), key=lambda x: abs((x[0][0] + x[0][1]) / 2 - trg_span[1]))]
                    arg_span2type=dict(zip(keys,values))

                    sentences_tk = sum(instance['sentences'], [])
                    doc=' '.join(sentences_tk)
                    doc = re.sub(r'[\u201c\u201d]', '\"', doc)
                    doc = re.sub(r'[\u2019\u2018]', '\'', doc)
                    doc = re.sub(r'[—–]', '-', doc)
                    sentences_tk = (doc).split(' ')
                    trigger_text = ' '.join(sentences_tk[trg_span[0]:trg_span[1] + 1])

                    documental_amrs = amrs[amr_id:amr_id + len(instance['sentences'])]
                    evt_key = documental_amrs[0].split('\n')[0]

                    if not evt_key.startswith('evt_mention'): ipdb.set_trace()
                    documental_amrs[0] = '\n'.join(documental_amrs[0].split('\n')[1:])
                    tokens = documental_amrs[0].split('\n')[0]
                    if not tokens.startswith('# ::tok '):
                        if tokens.startswith('WrongSample'):
                            amr_id +=1
                            while not amrs[amr_id].startswith('evt_mention'):
                                amr_id+=1
                            err_doc += 1
                            continue
                        else:
                            ipdb.set_trace()

                    tokens = tokens[len("# ::tok "):].split(' ')
                    if tokens[:min(5,len(tokens)-1)]!=sentences_tk[:min(5,len(tokens)-1)] or 'clusters' not in instance: # <ROOT> <trg> # or
                        if 'clusters' in instance:
                            ipdb.set_trace()
                            print(tokens[:7],sentences_tk[:7])
                            amr_id += len(json.loads(all_dict[i - 1])['sentences'])  # due to the errs in amr parser are solved by copying previous doc's amr, so we + previous len(sentences)
                            # ipdb.set_trace()
                        else:
                            amr_id += len(instance['sentences'])
                        err_doc+=1
                        continue

                    amr_id += len(instance['sentences'])

                    graph = load_amrs_to_graph(documental_amrs)
                    graph.merge_coref({"clusters": instance['clusters'], 'document': instance['document']})
                    if not graph.tokens[0][0] == sentences_tk[0]: ipdb.set_trace()
                if kwargs['path_method']=='all_T2As':
                    for links in instance['gold_evt_links']:
                        arg_span = links[1]
                        arg_text = ' '.join(sentences_tk[arg_span[0]:arg_span[1] + 1])
                        arg_type = re.split('[\d]+', links[2])[-1]
                        path = graph.BFS((trg_span[0], trg_span[1] + 1), (arg_span[0], arg_span[1] + 1))


                        if path == None or len(path) == 1:
                            err += 1
                            continue

                        path_text,intersentence,begin,end=self.print_path(path,sentences_tk)
                        path_text=' '.join(path_text)
                        example = InputExample(guid=str(i) + '.' + split,label=1,
                                               text_a=' '.join(sentences_tk).replace(trigger_text,'<trg>'+trigger_text+'<trg>'),
                                               tgt_text= path_text.replace(begin,begin+' '+arg_type),
                                               text_b=path_text.replace(begin,"<arg>"+' '+arg_type),
                                               meta={'event_type':event_type, 'trigger_text': trigger_text,
                                                     'arg_text': arg_text, 'arg_type': arg_type,
                                                     'trigger_span': trg_span, 'arg_span': arg_span,
                                                     'path_begin': ' '.join(begin), 'path_end': ' '.join(end)
                                                     ,'doc_key': instance['doc_key'],})
                        examples.append(example)

                elif kwargs['path_method']=='SteinerTree': # too many bugs in this Lib
                    NotFound = True
                    if trigger_text != instance['trigger_text']: continue
                    # arg_span_1st as the nearest
                    arg_span_1st=list(arg_span2type.keys())[0]
                    NotFound=False

                    if NotFound:
                        if instance['gold_evt_links']!=[]:
                            arg_span_1st=list(arg_span2type.keys())[0]
                        else:
                            err_doc += 1
                            continue

                    A2ApathExist = False
                    for arg_span in list(arg_span2type.keys()):
                        if arg_span==arg_span_1st: continue
                        try:
                            steinerPath=graph.steinerTree(trigger_span=(trg_span[0], trg_span[1]+1), clue_arg_span=(arg_span_1st[0], arg_span_1st[1] + 1),
                                              target_arg_span=(arg_span[0], arg_span[1] + 1))
                        except:
                            ipdb.set_trace()
                            graph.steinerTree(trigger_span=(trg_span[0], trg_span[1]+1), clue_arg_span=(arg_span_1st[0], arg_span_1st[1] + 1),
                                              target_arg_span=(arg_span[0], arg_span[1] + 1))
                        if steinerPath != None and len(steinerPath) != 1:
                            A2ApathExist=True
                            steinerPath_text, intersentence, _ , _ = self.print_path(steinerPath, sentences_tk)
                            arg_text=' '.join(sentences_tk[graph.find_substitute(arg_span)[0],graph.find_substitute(arg_span)[1]])
                            arg_1st_text=' '.join(sentences_tk[graph.find_substitute(arg_span_1st)[0],graph.find_substitute(arg_span_1st)[1]])
                            steinerPath_text=' '.join(steinerPath_text).replace(arg_text,arg_text+' '+ arg_span2type[arg_span])
                            steinerPath_text=steinerPath_text.replace(arg_1st_text,arg_1st_text+' '+ arg_span2type[arg_span_1st])
                            ipdb.set_trace()
                            maskedsteinerPath_text=steinerPath_text.replace(arg_text,'<arg>')
                            # full_masked_path=maskedT2Apath_text+' ' +maskedA2Apath_text
                            example = InputExample(guid=str(i) + '.' + split, label=1,
                                                   text_a=' '.join(sentences_tk).replace(trigger_text,
                                                                                         '<trg>' + trigger_text + '<trg>'),
                                                   tgt_text=steinerPath_text,
                                                   text_b=maskedsteinerPath_text,
                                                   meta={'event_type': event_type, 'trigger_text': trigger_text,
                                                         'trigger_span': trg_span, 'arg_1st':arg_1st,
                                                         'arg_2nd':arg_text,'doc_key': instance['doc_key'],
                                                         'evt_key':instance['evt_key'] # denote if dataset==RAMS
                                                         })
                            examples.append(example)

                elif kwargs['path_method']=='T2A2A':
                    NotFound = True
                    # if trigger_text != instance['trigger_text']: continue
                    # arg_span_1st as the nearest
                    if instance['gold_evt_links']!=[]:
                        arg_span_1st=list(arg_span2type.keys())[0]
                        NotFound=False

                    if NotFound:
                        if instance['gold_evt_links']!=[]:
                            arg_span_1st=list(arg_span2type.keys())[0]
                        else:
                            err_doc += 1
                            continue

                    T2Apath = graph.BFS((trg_span[0],trg_span[1] + 1), (arg_span_1st[0], arg_span_1st[1] + 1))
                    if T2Apath==None or len(T2Apath) == 1:
                        # err_doc += 1
                        # generate splited paths , useless
                        # for arg_span in list(arg_span2type.keys()):
                        #     T2Apath = graph.BFS((trg_id, trg_id + 1), (arg_span[0], arg_span[1] + 1))
                        #     if T2Apath == None or len(T2Apath) == 1:
                        #         err+=1
                        #         continue
                        #     T2Apath_text, intersentence, arg_1st, end = self.print_path(T2Apath, sentences_tk)
                        #     T2Apath_text = ' '.join(T2Apath_text).replace(arg_1st,
                        #                                                   arg_1st + ' ' + arg_span2type[arg_span])
                        #     example = InputExample(guid=str(i) + '.' + split, label=1,
                        #                            text_a=' '.join(sentences_tk).replace(trigger_text,
                        #                                                                  '<trg>' + trigger_text + '<trg>'),
                        #                            tgt_text=T2Apath_text,
                        #                            text_b=T2Apath_text.replace(arg_1st, '<arg>'),
                        #                            meta={'event_type': event_type, 'trigger_text': trigger_text,
                        #                                  'trigger_span': [trg_id, trg_id], 'arg_1st': arg_1st,
                        #                                 'doc_key': instance['doc_key'], })
                        #     examples.append(example)
                        continue

                    else:
                        T2Apath_text,intersentence,arg_1st,end = self.print_path(T2Apath,sentences_tk)
                        T2Apath_text=' '.join(T2Apath_text).replace(arg_1st,arg_1st+' '+ arg_span2type[arg_span_1st])
                        # pred arg_1st
                        # maskedT2Apath_text=T2Apath_text.replace(arg_1st,'<arg>')

                        A2ApathExist = False
                        for arg_span in list(arg_span2type.keys()):
                            if arg_span==arg_span_1st: continue
                            A2Apath = graph.BFS((arg_span_1st[0], arg_span_1st[1] + 1), (arg_span[0], arg_span[1] + 1))
                            if A2Apath != None and len(A2Apath) != 1:
                                A2ApathExist=True
                                A2Apath_text, intersentence, arg_text, end = self.print_path(A2Apath, sentences_tk)
                                full_path = T2Apath_text
                                for text in A2Apath_text:
                                    if text not in T2Apath_text:
                                        full_path += ' ' + text
                                full_path = full_path.replace(arg_text, arg_text + ' ' + arg_span2type[arg_span])
                                full_masked_path=full_path.replace(arg_text, '<arg>')
                                example = InputExample(guid=str(i) + '.' + split, label=1,
                                                       text_a=' '.join(sentences_tk).replace(trigger_text,
                                                                                             '<trg>' + trigger_text + '<trg>'),
                                                       tgt_text=full_path,
                                                       text_b=full_masked_path,
                                                       meta={'event_type': event_type, 'trigger_text': trigger_text,
                                                             'trigger_span': trg_span, 'arg_1st':arg_1st,
                                                             'arg_2nd':arg_text,'doc_key': instance['doc_key'],
                                                             # 'evt_key':instance['evt_key'] # denote if dataset==RAMS
                                                             })
                                examples.append(example)
                            else:
                                # expand to 6991 samples
                                T2Apath = graph.BFS((trg_span[0], trg_span[1]+1), (arg_span[0], arg_span[1] + 1))
                                if T2Apath == None or len(T2Apath) == 1:
                                    err += 1
                                    continue
                                new_T2Apath_text, intersentence, arg_text, end = self.print_path(T2Apath, sentences_tk)
                                full_path=T2Apath_text
                                for text in new_T2Apath_text:
                                    if text not in T2Apath_text:
                                        full_path+=' '+text
                                full_path=full_path.replace(arg_text,arg_text + ' ' + arg_span2type[arg_span])

                                example = InputExample(guid=str(i) + '.' + split, label=1,
                                                       text_a=' '.join(sentences_tk).replace(trigger_text,
                                                                                             '<trg>' + trigger_text + '<trg>'),
                                                       tgt_text=full_path, # TODO: how does it work?
                                                       text_b=full_path.replace(arg_text, '<arg>'),
                                                       meta={'event_type': event_type, 'trigger_text': trigger_text,
                                                             'trigger_span': trg_span, 'arg_1st': arg_1st,
                                                             'arg_2nd': arg_text
                                                           , 'doc_key': instance['doc_key'], })
                                examples.append(example)

                elif kwargs['path_method']=='maskedT2As':
                    for links in instance['gold_evt_links']:
                        arg_span = links[1]
                        # if trigger_span==arg_span: continue
                        trigger_text = ' '.join(sentences_tk[trg_span[0]:trg_span[1] + 1])
                        arg_text = ' '.join(sentences_tk[arg_span[0]:arg_span[1] + 1])
                        arg_type = re.split('[\d]+', links[2])[-1]
                        path = graph.BFS((trg_span[0], trg_span[1] + 1), (arg_span[0], arg_span[1] + 1))

                        if path == None or len(path) == 1:
                            err += 1
                            continue

                        path_text,intersentence,begin,end=self.print_path(path,sentences_tk)

                        raw_path_text=' '.join(path_text)
                        mask_poses=random.choices(path_text,k=1)
                        masked_path_text=raw_path_text
                        for mask in mask_poses:
                            if mask==begin:continue
                            masked_path_text=masked_path_text.replace(mask,'<arg>')

                        example = InputExample(guid=str(i) + '.' + split,label=1,
                                               text_a=' '.join(sentences_tk).replace(trigger_text,'<trg>'+trigger_text+'<trg>'),
                                               tgt_text= raw_path_text.replace(begin,begin+' '+arg_type),
                                               text_b=masked_path_text.replace(begin,"<arg>"+' '+arg_type),
                                               meta={'event_type':event_type, 'trigger_text': trigger_text,
                                                     'arg_text': arg_text, 'arg_type': arg_type,
                                                     'trigger_span': trg_span, 'arg_span': arg_span,
                                                     'path_begin': ' '.join(begin), 'path_end': ' '.join(end)
                                                     ,'doc_key': instance['doc_key'],})
                        examples.append(example)


                if (i + 1) % 500 == 0:
                    print(f'have handled {i + 1} documents')

        print('err num is ', err, f',err doc num is {err_doc}, ttl number of examples is ', len(examples))

        return examples



if __name__=='__main__':
    examples=WikiEventProcesser().get_examples(data_dir='./data/wikievents/informative', split='train', use_info=True,
                                               ontology_file='event_role_wikievents.json',data_part='full',recordAsRAMS=True)
    exampleInspection=[]
    ttl_sent=0
    for ex in examples:
        instance=ex.meta
        exampleInspection.append(instance)
    with open('./data/wikievents/informative/train.jsonl','w') as f:
        f.writelines((json.dumps(ex)+'\n' for ex in exampleInspection))


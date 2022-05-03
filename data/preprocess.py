import json
import re

not_aligned=0
with open('./informative/train.jsonl') as rawf ,\
open (f'./WikiwCoref/corefered.json') as cff:
    rawflines=rawf.readlines()
    cfflines=cff.readlines()
    assert len(rawflines)==len(cfflines)
    examples_with_coref=''
    for i in range(len(rawflines)):
        instance=json.loads(rawflines[i])
        coref_info=json.loads(cfflines[i])
        coref_doc=' '.join(coref_info['document'])
        coref_doc = re.sub(r'[\u201c\u201d]', '\"', coref_doc)
        coref_doc = re.sub(r'[\u2019\u2018]', '\'', coref_doc)
        coref_doc = re.sub(r'[—–]', '-', coref_doc)
        instance_doc=' '.join(sum(instance['sentences'],[]))
        instance_doc = re.sub(r'[\u201c\u201d]', '\"', instance_doc)
        instance_doc = re.sub(r'[\u2019\u2018]', '\'', instance_doc)
        instance_doc = re.sub(r'[—–]', '-', instance_doc)
        if  instance_doc==coref_doc :
            instance['document']=coref_doc.split(' ')
            instance['clusters']=coref_info['clusters']
        else : not_aligned+=1

        for sent_i in range(len(instance['sentences'])):
            sent=' '.join(instance['sentences'][sent_i])
            sent = re.sub(r'[\u201c\u201d]', '\"', sent)
            sent = re.sub(r'[\u2019\u2018]', '\'', sent)
            sent = re.sub(r'[—–]', '-', sent)
            instance['sentences'][sent_i]=sent.split(' ')

        examples_with_coref+=(json.dumps(instance)+'\n')
    f=open(f'./WikiwCoref/informative/train.jsonl','w',encoding='utf8')
    f.write(examples_with_coref)
    f.close()

print(not_aligned,len(rawflines))
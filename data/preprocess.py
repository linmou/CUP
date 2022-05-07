import json
import re
import sys
import argparse

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--coref_dir',type=str,require=True)
    parser.add_argument('--train_dir',type=str,required=True)
    parser.add_argument('--output_dir',type=str,required=True)
    args=parser.parse_args()
    not_aligned=0
    with open(args.train_dir) as rawf ,\
    open (args.coref_dir) as cff:
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
        f=open(args.output_dir,'w',encoding='utf8')
        f.write(examples_with_coref)
        f.close()

    print(not_aligned,len(rawflines))
import argparse
import ipdb
import numpy as np

import torch
from DataProcessers import RAMSProcesser,AmrGraphProcesser,WikiEventProcesser
from scorer import *

from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from openprompt.prompts import SoftTemplate,ManualTemplate
from openprompt.plms import load_plm,add_special_tokens
from openprompt import PromptDataLoader,PromptForGeneration ,PromptForConstrainedGeneration
from openprompt.utils.metrics import generation_metric

from itertools import cycle
import random
from tensorboardX import SummaryWriter
import log
logger = log.get_logger("root")

generation_arguments = {
    "max_length": 50,
    "max_new_tokens": None,
    "min_length": 5,
    "temperature": 1.0,
    "do_sample": False,
    "top_k": 0,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "num_beams": 1,
    "bad_words_ids": None
}


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # if args.cuda:
    torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True

def eval_path_inference(prompt_model,dataloader)->dict:
    generated_sentence = []
    groundtruth_sentence = []
    for step, inputs in enumerate(dataloader):
        if args.cuda:
            inputs = inputs.cuda()
        if 'Constrained' in args.model:
            _,output_sentences = prompt_model.generate(**inputs, **generation_arguments)
        else:
            _, output_sentences = prompt_model.generate(inputs, **generation_arguments)
        groundtruth_sentence.extend(inputs['tgt_text'])
        generated_sentence.extend(output_sentences)
    if len(generated_sentence)<=5:
        print(generated_sentence)
        print(groundtruth_sentence)
    score = sum([ 1 if gen == gold else 0 for gen,gold in zip(generated_sentence, groundtruth_sentence) ])/len(groundtruth_sentence)
    print("test_score", score, flush=True)
    return {"absolute_score":score}

def evaluate(prompt_model,dataloader,pred_only=False): #
    prompt_model.eval()
    import time
    predicts=[]
    start_time=time.time()
    for step, inputs in enumerate(dataloader):
        if args.cuda:
            inputs = inputs.cuda()
        doc_keys=[raw_data.meta['doc_key'] for raw_data in dataloader.raw_dataset[step*args.eval_batchsize:(step+1)*args.eval_batchsize]]
        if 'Constrained' in args.model:
            _,output_sentences = prompt_model.generate(**inputs, **generation_arguments)
        else:
            _, output_sentences = prompt_model.generate(inputs, **generation_arguments)
        predicts.extend([{'doc_key':doc_key,'predicted':out, 'gold':tgt} for (doc_key,out,tgt) in zip(doc_keys, output_sentences, inputs['tgt_text'])])
    gen_consumption= time.time()-start_time
    if pred_only: return predicts
    if args.dataset=='RAMS':
        outputs = covert_gen2outs(args,predicts,dataloader.raw_dataset)
        # print('target:',predicts[0]['gold'],'\nprediction:',predicts[0]['predicted'])
        return_dict=run_evaluation(args,outputs)
    elif args.dataset=='wikievents':
        # print('target:',predicts[0]['gold'],'\nprediction:',predicts[0]['predicted'])
        return_dict=WikiEvaluate(args,predicts)
    logger.info("eval_score"+json.dumps( return_dict['metrics']))
    return return_dict


def pipeline_decode(args,prompt_model,template,tokenizer,WrapperClass,split:str='test',method:str='fine_grained'):
    prompt_model.eval()
    print(f'start {method} pipeline evaluation')
    # predict intra sent arg
    if args.dataset == 'RAMS':
        Processer= RAMSProcesser()
        ontology_file='aida_ontology_cleaned.csv'
        # ontology_file='wikievents_ontology.csv' # when dis impact test

    elif args.dataset == 'wikievents':
        Processer= WikiEventProcesser()
        ontology_file='event_role_wikievents.json'
    else:
        raise NotImplemented
    if method in ['coarse_grained','fine_grained']:
        test_dataset =Processer.get_examples(args.data_path, split=split,
                                                                 data_part='easy',use_info=False,
                                                                 ontology_file=ontology_file,
                                                                 k_shot=args.num_dev_samples)

        test_dataloader = PromptDataLoader(dataset=test_dataset, template=template, tokenizer=tokenizer,
                                           tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_len,
                                           decoder_max_length=args.decoder_max_len,
                                           batch_size=args.eval_batchsize, shuffle=False, teacher_forcing=False,
                                           predict_eos_token=False,
                                           truncate_method="tail")

        easy_predicts=evaluate(prompt_model,test_dataloader,pred_only=True)

    if method == 'coarse_grained':
        # predict inter sent arg based on the prediction of intra sent args
        test_dataset = Processer.get_examples(args.data_path, split=split,
                                                                 data_part='full',use_info=False,
                                                                ontology_file=ontology_file,
                                                                 k_shot=args.num_dev_samples)
        assert len(easy_predicts)==len(test_dataset)
        for insid,instance in enumerate(test_dataset):
            if not easy_predicts[insid]['doc_key']==instance.meta['doc_key']:ipdb.set_trace()
            instance.text_b=easy_predicts[insid]['predicted']

        test_dataloader = PromptDataLoader(dataset=test_dataset, template=template, tokenizer=tokenizer,
                                           tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_len,
                                           decoder_max_length=args.decoder_max_len,
                                           batch_size=args.eval_batchsize, shuffle=False, teacher_forcing=False,
                                           predict_eos_token=False,
                                           truncate_method="tail")

    elif method in ['fine_grained']:
        # Given2For13
        test_dataset = Processer.get_examples(args.data_path, split=split,
                                                    data_part='Given2For13',use_info=False,
                                                    ontology_file=ontology_file,
                                                    k_shot=args.num_dev_samples)

        assert len(easy_predicts) == len(test_dataset)
        for insid, instance in enumerate(test_dataset):
            if not easy_predicts[insid]['doc_key'] == instance.meta['doc_key']: ipdb.set_trace()
            instance.text_b = easy_predicts[insid]['predicted']


        test_dataloader = PromptDataLoader(dataset=test_dataset, template=template, tokenizer=tokenizer,
                                           tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_len,
                                           decoder_max_length=args.decoder_max_len,
                                           batch_size=args.eval_batchsize, shuffle=False, teacher_forcing=False,
                                           predict_eos_token=False,
                                           truncate_method="tail")
        harder_predicts = evaluate(prompt_model, test_dataloader, pred_only=True)

        # Given123For04
        test_dataset = Processer.get_examples(args.data_path, split=split,
                                                    data_part='Given123For04',use_info=False,
                                                    ontology_file=ontology_file,
                                                    k_shot=args.num_dev_samples)
        assert len(harder_predicts) == len(test_dataset)
        for insid, instance in enumerate(test_dataset):
            if not harder_predicts[insid]['doc_key'] == instance.meta['doc_key']: ipdb.set_trace()
            instance.text_b = harder_predicts[insid]['predicted']

        test_dataloader = PromptDataLoader(dataset=test_dataset, template=template, tokenizer=tokenizer,
                                           tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_len,
                                           decoder_max_length=args.decoder_max_len,
                                           batch_size=args.eval_batchsize, shuffle=False, teacher_forcing=False,
                                           predict_eos_token=False,
                                           truncate_method="tail")

    return evaluate(prompt_model,test_dataloader)

def setup_all(args):
    """
    set up plm, data, and prompt
    """
    dataset = {}
    if not args.eval_only:
        if args.dataset=='RAMS':
            dataset['train'] = RAMSProcesser().get_train_examples(args.data_path,data_part=args.data_part,
                                                              ontology_file=args.ontology_file,k_shot=args.num_train_samples)
        elif args.dataset=='wikievents':
            dataset['train'] = WikiEventProcesser().get_train_examples(args.data_path,use_info=False, ontology_file='event_role_wikievents.json',data_part=args.data_part
                                                                  )

        if args.augment:
            if args.dataset=='wikievents':
                dataset['AmrGraph_train'] = AmrGraphProcesser().get_train_examples(
                                                                                './data/wikievents/WikiwCoref/informative', #
                                                                               amr_path=f'./data/{args.dataset}/amr/train.amr.txt',dataset=args.dataset,
                                                                               ontology_file='wikievents_ontology.csv',path_method=args.path_method,)
            elif args.dataset=='RAMS':
                dataset['AmrGraph_train'] = AmrGraphProcesser().get_train_examples(
                    args.data_path,  #
                    amr_path=f'./data/{args.dataset}/amr/train.amr.txt', dataset=args.dataset,
                    ontology_file='aida_ontology_cleaned.csv', path_method=args.path_method, )

    if not args.train_only:
        if args.dataset == 'RAMS':
            dataset['validation'] = RAMSProcesser().get_dev_examples(args.data_path, amr_path='./data/RAMS/amr/dev.amr.txt',data_part=args.data_part,
                                                                     ontology_file=args.ontology_file,k_shot=args.num_dev_samples)
            dataset['test'] = RAMSProcesser().get_test_examples(args.data_path, amr_path='./data/RAMS/amr/test.amr.txt',data_part=args.data_part,
                                                                ontology_file=args.ontology_file,k_shot=args.num_test_samples)
        elif args.dataset == 'wikievents':
            dataset['validation'] = WikiEventProcesser().get_dev_examples(args.data_path, use_info=False, ontology_file='event_role_wikievents.json',data_part=args.data_part)
            dataset['test'] = WikiEventProcesser().get_test_examples(args.data_path, use_info=False, ontology_file='event_role_wikievents.json',data_part=args.data_part)



    plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path, )# args.model_name_or_path args.ckptPath
    tokenizer.add_tokens(['<arg>', '<trg>'])
    plm.resize_token_embeddings(len(tokenizer))

    RAMSTemplate = ManualTemplate(tokenizer=tokenizer,
                                  text='{"placeholder":"text_b"} {"special": "<eos>"} {"placeholder":"text_a"} {"special": "<eos>"} {"mask"}')
    # AmrGraphTemplate = ManualTemplate( tokenizer=tokenizer,
    #                                 text='{"placeholder":"text_b"} {"special": "<eos>"} {"placeholder":"text_a"} {"special": "<eos>"} {"mask"}')
    #
    # RAMSTemplate = SoftTemplate(model=plm, tokenizer=tokenizer, text=' {"placeholder":"text_b"} {"special": "<eos>"} {"soft"} {"placeholder":"text_a"} {"special": "<eos>"} {"mask"}.',num_tokens=80)
    #
    # AmrGraphTemplate = SoftTemplate(model=plm, tokenizer=tokenizer, text='{"soft"} {"placeholder":"text_b"} {"special": "<eos>"} {"placeholder":"text_a"} {"special": "<eos>"} {"mask"}.',num_tokens=80)

    # RAMSTemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text='{"placeholder":"text_a"} {"soft": "Question:"} {"placeholder":"text_b"}? Is it correct? {"mask"}.')

    amr_dataloader,train_dataloader = None,None
    if not args.eval_only: # visualize one instance for checking
        if args.augment :
            wrapped_example = RAMSTemplate.wrap_one_example(dataset['AmrGraph_train'][0])
        else:
            wrapped_example = RAMSTemplate.wrap_one_example(dataset['train'][0])
        print(wrapped_example)

        train_dataloader = PromptDataLoader(dataset=dataset["train"], template=RAMSTemplate, tokenizer=tokenizer,
                                            tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_len,
                                            decoder_max_length=args.decoder_max_len,k_shot=args.num_train_samples,
                                            batch_size=args.train_batchsize, shuffle=True, teacher_forcing=True,
                                            predict_eos_token=True,
                                            truncate_method="tail")

        if args.augment:
            amr_dataloader = PromptDataLoader(dataset=dataset["AmrGraph_train"], template=RAMSTemplate, tokenizer=tokenizer,
                                              tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_len,
                                              decoder_max_length=args.decoder_max_len,
                                              batch_size=args.train_batchsize, shuffle=True, teacher_forcing=True, predict_eos_token=True,
                                              truncate_method="tail")

    validation_dataloader,test_dataloader=None,None
    if not args.train_only:
        validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=RAMSTemplate, tokenizer=tokenizer,
                                                tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_len,
                                                decoder_max_length=args.decoder_max_len,
                                                batch_size=args.eval_batchsize, shuffle=False, teacher_forcing=False,
                                                predict_eos_token=False,
                                                truncate_method="tail")

        test_dataloader = PromptDataLoader(dataset=dataset["test"], template=RAMSTemplate, tokenizer=tokenizer,
                                           tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_len,
                                           decoder_max_length=args.decoder_max_len,
                                           batch_size=args.eval_batchsize, shuffle=False, teacher_forcing=False,
                                           predict_eos_token=False,
                                           truncate_method="tail")
    return plm, tokenizer, model_config, WrapperClass,RAMSTemplate,dataset,train_dataloader,amr_dataloader,validation_dataloader,test_dataloader


def train(args,plm, tokenizer, model_config, WrapperClass,template,dataset,train_dataloader,amr_dataloader,validation_dataloader,test_dataloader):
    training_record_path=os.path.join(args.storage_dir,f'{template.name}_{args.model}_lr{args.lr}/training_process')
    if not os.path.exists(training_record_path):
        os.makedirs(training_record_path)
    writer = SummaryWriter(training_record_path)

    global_step = 0
    tot_loss = 0
    log_loss = 0
    # training and generation.
    if 'Constrained' in args.model:
        prompt_model = PromptForConstrainedGeneration(plm=plm, template=template, freeze_plm=False, tokenizer=tokenizer,)
    else:
        prompt_model = PromptForGeneration(plm=plm, template=template, freeze_plm=False, tokenizer=tokenizer,)

    if args.cuda:
        prompt_model = prompt_model.cuda()


    if args.eval_only:
        prompt_model.load_state_dict(torch.load(os.path.join(args.ckptPath,'pytorch_model.bin')),strict=False)
        logger.info('model loaded')
        args.gold_file = os.path.join(args.data_path, 'test.jsonl')
        if args.pipeline_decode:
            result=pipeline_decode(args, prompt_model, template, tokenizer, WrapperClass,split='test',method='fine_grained')
        else:
            result=evaluate(prompt_model, test_dataloader)
        with open(os.path.join(args.ckptPath, 'results.json'), 'w') as f:
            f.write(json.dumps(result))
        from pprint import pprint
        pprint(result)
    else:
        if args.train_on_pretrained:
            prompt_model.load_state_dict(torch.load(os.path.join(args.ckptPath, 'pytorch_model.bin')))
            # frozen_modules=prompt_model.plm.model.encoder.layers[-5:]
            # for module in frozen_modules:
            #     for param in module.parameters():
            #         param.requires_grad=False
        if not args.train_only:
            args.gold_file = os.path.join(args.data_path, 'test.jsonl')
            if args.pipeline_decode:
                pipeline_decode(args, prompt_model, template, tokenizer, WrapperClass)
            else:
                evaluate(prompt_model,test_dataloader)

        optimizer_grouped_parameters = [
            {'params': [p for n, p in prompt_model.named_parameters() if "raw_embedding" not in n and p.requires_grad]}
        ]


        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        tot_step = len(train_dataloader) * (args.numEpoch) * 2
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)
        best_eval_f1=0


        for epoch in tqdm(range(args.numEpoch)):
            if args.augment:
                if len(train_dataloader)<len(amr_dataloader):
                    train_itertor = zip(cycle(train_dataloader), (amr_dataloader))
                else:
                    train_itertor = zip((train_dataloader), cycle(amr_dataloader))
            else:
                train_itertor = train_dataloader

            for epoch_step, inputs in enumerate((train_itertor)):
                if args.augment:
                    amr_inputs=inputs[1]
                    inputs=inputs[0]

                if args.cuda:
                    inputs = inputs.cuda()
                    if args.augment:
                        amr_inputs = amr_inputs.cuda()

                global_step += 1
                prompt_model.train()
                if 'Constrained' in args.model: # constrained generation
                    loss = prompt_model(**inputs)
                    if args.augment:
                        loss_amr=prompt_model(**amr_inputs)
                        loss+=args.alpha*loss_amr
                else:
                    loss = prompt_model(inputs)
                    if args.augment:
                        loss_amr=prompt_model(amr_inputs)
                        loss+=args.alpha*loss_amr

                loss.backward()
                tot_loss += loss.item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if global_step % args.logStep == 0 :
                    avg_loss=(tot_loss - log_loss) / args.logStep
                    logger.info("Epoch {}, global_step {} average loss: {} lr: {}  ".format(epoch, global_step,avg_loss,
                                                                                    scheduler.get_last_lr()[0]))
                    writer.add_scalar('loss',(tot_loss - log_loss) / args.logStep,global_step)
                    log_loss = tot_loss
                    # eval in training
                    if epoch >=args.eval_start and (not args.train_only) :#and avg_loss<10
                        args.gold_file = os.path.join(args.data_path, 'dev.jsonl')
                        if args.pipeline_decode:
                            result = pipeline_decode(args, prompt_model, template, tokenizer, WrapperClass,
                                                     split='dev')
                        else:
                            result = evaluate(prompt_model,validation_dataloader )
                        writer.add_scalars('performance',result,global_step)
                        if result['metrics']['f1'] > best_eval_f1:
                            best_eval_f1 = result['metrics']['f1']
                            save_path = os.path.join(args.storage_dir,
                                                     '{}_{}_lr{}/E{}_S{}_P{:.2f}'.format(template.name, args.model,
                                                                                         args.lr,
                                                                                         epoch, global_step,
                                                                                         result['metrics']['f1']))
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            prompt_model.save(save_path)
                            save_args(args, save_path)
                            with open(os.path.join(save_path, 'results.json'), 'w') as f:
                                f.write(json.dumps(result))



            if not args.train_only: # eval at the end of each epoch
                args.gold_file = os.path.join(args.data_path,'test.jsonl')
                if args.pipeline_decode:
                    result = pipeline_decode(args, prompt_model, template, tokenizer, WrapperClass)
                else:
                    result = evaluate(prompt_model, validation_dataloader)

                if result['metrics']['f1'] > best_eval_f1:
                    best_eval_f1=result['metrics']['f1']
                    save_path = os.path.join(args.storage_dir,
                                             '{}_{}_lr{}/E{}_S{}_P{:.2f}'.format(template.name, args.model, args.lr,
                                                                                 epoch, global_step,result['metrics']['f1']))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    prompt_model.save(save_path)
                    save_args(args,save_path)
                    with open(os.path.join(save_path,'results.json'),'w') as f:
                        f.write(json.dumps(result))

            elif epoch>=1:
                save_path = os.path.join(args.storage_dir,
                                     '{}_{}_lr{}/E{}_S{}_L{:.2f}'.format(template.name, args.model, args.lr,
                                                                         epoch, global_step, tot_loss/global_step))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                prompt_model.save(save_path)
                save_args(args, save_path)

        return prompt_model


def save_args(args,path):
    args_dict=args.__dict__
    with open(os.path.join(path,'args.json'),'w') as f:
        json.dump(args_dict,f)

def curriculum_learnig(args):
    args.data_part='easy'
    args.augment=False
    args.train_only=True
    args.numEpoch=3
    plm, tokenizer, model_config, WrapperClass,template,dataset,train_dataloader,amr_dataloader,validation_dataloader,test_dataloader=setup_all(args)
    easy_train=dataset['train']
    prompt_model=train(args,plm, tokenizer, model_config, WrapperClass,template,dataset,train_dataloader,amr_dataloader,validation_dataloader,test_dataloader)


    args.data_part = 'Given2For13'
    args.lr = 5e-5
    args.numEpoch = 3
    args.train_only=True
    plm, tokenizer, model_config, WrapperClass, template, dataset, train_dataloader, amr_dataloader, validation_dataloader, test_dataloader = setup_all(
        args)
    harder_train = dataset['train']
    combined_train = harder_train + easy_train
    train_dataloader = PromptDataLoader(dataset=combined_train, template=template, tokenizer=tokenizer,
                                        tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_len,
                                        decoder_max_length=args.decoder_max_len, k_shot=args.num_train_samples,
                                        batch_size=args.train_batchsize, shuffle=True, teacher_forcing=True,
                                        predict_eos_token=True,
                                        truncate_method="tail")
    prompt_model = train(args, prompt_model.plm, tokenizer, model_config, WrapperClass, template, dataset,
                         train_dataloader,
                         amr_dataloader, validation_dataloader, test_dataloader)

    args.data_part = 'Given123For04'
    args.lr = 3e-5
    args.numEpoch = 4
    args.train_only = True
    args.augment=True
    args.path_method='T2A2A'
    args.train_batchsize=16
    args.alpha=0.7
    plm, tokenizer, model_config, WrapperClass, template, dataset, train_dataloader, amr_dataloader, validation_dataloader, test_dataloader = setup_all(
        args)
    combined_train = dataset['train'] + combined_train
    T2A2A_set=dataset['AmrGraph_train']
    train_dataloader = PromptDataLoader(dataset=combined_train, template=template, tokenizer=tokenizer,
                                        tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_len,
                                        decoder_max_length=args.decoder_max_len, k_shot=args.num_train_samples,
                                        batch_size=args.train_batchsize, shuffle=True, teacher_forcing=True,
                                        predict_eos_token=True,
                                        truncate_method="tail")
    prompt_model = train(args, prompt_model.plm, tokenizer, model_config, WrapperClass, template, dataset,
                         train_dataloader,
                         amr_dataloader, validation_dataloader, test_dataloader)

    save_path = os.path.join(args.storage_dir,
                             '{}_{}_lr{}/fine_grained_T2A2A'.format(template.name, args.model, args.lr, ))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    prompt_model.save(save_path)
    save_args(args, save_path)

    ###########################
    # args.data_part = 'Given2For0134'
    # args.lr=5e-5
    # args.train_only=True
    # args.augment=True
    # args.path_method='T2A2A'
    # args.train_batchsize=16
    # args.numEpoch=3
    # plm, tokenizer, model_config, WrapperClass, template, dataset, train_dataloader, amr_dataloader, validation_dataloader, test_dataloader = setup_all(
    #     args)
    # hard_train=dataset['train']
    # combined_train=hard_train+easy_train
    # train_dataloader = PromptDataLoader(dataset=combined_train, template=template, tokenizer=tokenizer,
    #                                     tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_len,
    #                                     decoder_max_length=args.decoder_max_len, k_shot=args.num_train_samples,
    #                                     batch_size=args.train_batchsize, shuffle=True, teacher_forcing=True,
    #                                     predict_eos_token=True,
    #                                     truncate_method="tail")
    #
    # prompt_model = train(args, prompt_model.plm, tokenizer, model_config, WrapperClass, template, dataset, train_dataloader,
    #                      amr_dataloader, validation_dataloader, test_dataloader)
    #
    # save_path = os.path.join(args.storage_dir,
    #                          '{}_{}_lr{}/coarse_grained_T2A2A'.format(template.name, args.model, args.lr,)) # f1=43
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # prompt_model.save(save_path)
    # save_args(args, save_path)
    ###################################

    args.data_part = 'full'
    args.numEpoch = 6
    args.train_only = False
    args.augment=True
    args.path_method='all_T2As'
    args.lr=2e-5
    args.train_batchsize=16
    plm,tokenizer, model_config, WrapperClass,template,dataset,train_dataloader,amr_dataloader,validation_dataloader,test_dataloader=setup_all(args)
    combined_train = dataset['train']+combined_train
    augment_set = dataset['AmrGraph_train']+T2A2A_set
    train_dataloader = PromptDataLoader(dataset=combined_train, template=template, tokenizer=tokenizer,
                                        tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_len,
                                        decoder_max_length=args.decoder_max_len, k_shot=args.num_train_samples,
                                        batch_size=args.train_batchsize, shuffle=True, teacher_forcing=True,
                                        predict_eos_token=True,
                                        truncate_method="tail")
    amr_dataloader = PromptDataLoader(dataset=augment_set, template=template, tokenizer=tokenizer,
                                        tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_len,
                                        decoder_max_length=args.decoder_max_len, k_shot=args.num_train_samples,
                                        batch_size=args.train_batchsize, shuffle=True, teacher_forcing=True,
                                        predict_eos_token=True,
                                        truncate_method="tail")

    prompt_model=train(args,prompt_model.plm, tokenizer, model_config, WrapperClass,template,dataset,train_dataloader,amr_dataloader,validation_dataloader,test_dataloader)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # args for dataloader
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, default='data/wikievents/informative')
    parser.add_argument("--augment",type=bool,default=False,help='if use AmrGraph or other training set for multitask training')
    parser.add_argument("--max_seq_len",type=int,default=256,help='max_seq_len of prompt datalodaer')
    parser.add_argument("--decoder_max_len",type=int,default=128,help='max_decode_len of prompt datalodaer')
    parser.add_argument("--path_method",type=str,default='all_T2As')
    parser.add_argument("--data_part",type=str,default='easy')
    parser.add_argument("--dataset",type=str,default='wikievents',choices=['RAMS','wikievents'],)
    # args for train and eval
    parser.add_argument("--lr", type=float, default=1e-4) # 1e-5 for large ,1e-4 for base
    parser.add_argument("--numEpoch",type=int,default=6)
    parser.add_argument("--logStep",type=int,default=250)
    parser.add_argument("--alpha",type=float,default=0.7)
    parser.add_argument("--CL",action='store_true',default=False,help='whether use curriculum learning')
    parser.add_argument("--pipeline_decode",action='store_true',default=False)
    parser.add_argument("--eval_start",type=int,default=4)
    parser.add_argument("--train_batchsize",type=int,default=32)
    parser.add_argument("--eval_batchsize",type=int,default=32)
    parser.add_argument("--model", type=str, default='bart')
    parser.add_argument("--model_name_or_path", default='../pretrained_models/'
                                                        'bart-base')
    parser.add_argument("--plm_eval_mode", action="store_true")
    parser.add_argument("--cuda",default=True)
    parser.add_argument("--storage_dir",type=str,default='experiments/wikiE/original_info/random')
    parser.add_argument("--eval_only",action="store_true",default=False)
    parser.add_argument("--train_only",action="store_true",default=False)
    parser.add_argument("--train_on_pretrained",action="store_true",default=False)
    parser.add_argument("--ckptPath",type=str,default= 'experiments/wikiE/original_info/CL/CUP/Manual_bart_lr2e-05/E3_S900_P47.05')
    parser.add_argument("--num_train_samples",type=int)
    parser.add_argument("--num_dev_samples",type=int)
    parser.add_argument("--num_test_samples",type=int)

    # args for convert gen2out
    parser.add_argument('--test-file', type=str,default='/wikievents/informative/test.jsonl')
    parser.add_argument('--output-file',type=str, default='./experiments/test_output.jsonl')
    parser.add_argument('--ontology_file',type=str, default='wikievents_ontology.csv')
    parser.add_argument('--head-only',action='store_true',default=False)
    parser.add_argument('--coref', action='store_true', default=False)
    parser.add_argument('--coref-file', type=str,)
    parser.add_argument('--writedown', action='store_true', default=False,help='whether write outputs to a file')
    # args = parser.parse_args()
    # args for scorer
    parser.add_argument('-g', '--gold_file', type=str,#default=args.test_file,
                        help='Gold file path. In common sense, it is test-file')
    parser.add_argument('-p', '--pred_file', type=str, default=None,
                        help='Predictions file path. For special cases.')
    parser.add_argument('--reuse_gold_format', dest='reuse_gold_format',
                        default=True, action='store_true',
                        help="Reuse gold file format for pred file.")
    parser.add_argument('-cd', '--type_constrained_decoding', dest="cd",
                        default=False, action='store_true',
                        help="Use type constrained decoding" +
                             '(only possible when ontology file is given')
    parser.add_argument('--constraints_file', default="./event_role_multiplicities.txt",dest='constraints_file',
                        help="ontology file for type constrained decoding" )
    parser.add_argument('--do_all', dest='do_all', default=False,
                        action='store_true', help="Do everything.")
    parser.add_argument('--metrics', dest='metrics', default=False,
                        action='store_true',
                        help="Compute overall p, r, f1.")
    parser.add_argument('--distance', dest='distance', default=False,
                        action='store_true',
                        help="Compute p, r, f1 by distance.")
    parser.add_argument('--role_table', dest='role_table', default=False,
                        action='store_true',
                        help="Compute p, r, f1 per role.")
    parser.add_argument('--confusion', dest='confusion', default=False,
                        action='store_true',
                        help="Compute an error confusion matrix.")
    args = parser.parse_args()
    args.gold_file = os.path.join(args.data_path, 'test.jsonl')
    setup_seed(args.seed)
    if args.CL==True:
        curriculum_learnig(args)
    else:
        plm, tokenizer, model_config, WrapperClass,template,dataset,train_dataloader,amr_dataloader,validation_dataloader,test_dataloader=setup_all(args)
        # for alpha in [0.2, 0.4, 0.6, 0.8, 1]:  # find best alpha
        #     args.alpha = alpha
        #     for lr in [args.lr]:  # find the best lr by this loop
        #         args.lr=lr
        prompt_model=train(args,plm, tokenizer, model_config, WrapperClass,template,dataset,train_dataloader,amr_dataloader,validation_dataloader,test_dataloader)




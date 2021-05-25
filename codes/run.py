#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader

from model import KGEModel

from dataloader import TrainDataset, RuleTriple
from dataloader import BidirectionalOneShotIterator,Iterator

from rule_reasoning import one_time_reasoning
from collections import defaultdict
def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')

    parser.add_argument('-ur', '--use_rule', action='store_true')
    parser.add_argument('-urc', '--use_rule_coefficient', default=0.0001, type=float)
    parser.add_argument('-ntp', '--new_triple_path', type=str)
    parser.add_argument('--sparsity', action='store_true')
    parser.add_argument('--NNE', action='store_true')
    parser.add_argument('-cbs', '--conclusion_batch_size', default=32, type=int)
    args.max_steps_per('--max_steps_per', default=40000, type=int)

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    
    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None, 
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')
    
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')


    
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=300000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    
    return parser.parse_args(args)

def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']

    
def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )
    
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'), 
        entity_embedding
    )
    
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'), 
        relation_embedding
    )

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def read_triple2(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, t, r = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        if metric == 'rule triple score:':
            logging.info('%s %s at step %d: \n %s' % (mode, metric, step, str(metrics[metric])))
        else:
            logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
        
        
def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')
    
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')
    
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # Write logs to checkpoint and console
    set_logger(args)




    rules_path = './datasets/DB100K/Rules.txt'
    rules = []
    with open(rules_path, 'r', encoding='utf8') as f:
        for line in f:
            rule, p = line.strip().split('\t')
            rules.append(tuple([tuple(rule.split(',')), float(p)]))


    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        id2entity = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)
            id2entity[int(eid)] = entity

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        id2relation = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
            id2relation[int(rid)] = relation
    
    # Read regions for Countries S* datasets
    if args.countries:
        regions = list()
        with open(os.path.join(args.data_path, 'regions.list')) as fin:
            for line in fin:
                region = line.strip()
                regions.append(entity2id[region])
        args.regions = regions

    nentity = len(entity2id)
    nrelation = len(relation2id)
    
    args.nentity = nentity
    args.nrelation = nrelation
    
    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    
    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)

    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))
    
    #All true triples
    all_true_triples = train_triples + valid_triples + test_triples
    if args.sparsity:
        sparse_triples = read_triple2(os.path.join(args.data_path, 'test_sparsity_0.995.txt'), entity2id, relation2id)
        all_true_triples = train_triples + valid_triples + sparse_triples






    def conclusion_filtering(entity_embedding, relation_embedding, new_triple):

        def ComplEx_score(head, relation, tail):
            re_head, im_head = torch.chunk(head, 2, dim=0)
            re_relation, im_relation = torch.chunk(relation, 2, dim=0)
            re_tail, im_tail = torch.chunk(tail, 2, dim=0)
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail
            score = score.sum(dim=0)
            return score
        true_conclusions = []
        for h,r,t in new_triple:
            head, relation, tail = entity_embedding[h], relation_embedding[r], entity_embedding[t]
            if ComplEx_score(head, relation, tail)>0.99:
                true_conclusions.append((h,r,t))
        return true_conclusions

    def more_triple_p(new_triple_path):
        rule_num = 0
        rule_triples_index = []
        new_triple_only = []
        with open(new_triple_path) as f:
            for line in f:
                rule_num += 1
                rule_triples = line.strip().split('\t')
                p = float(rule_triples[0])
                rule_triples = rule_triples[1:]

                for i in rule_triples:
                    h, r, t = i.split('@$')
                    # h, r, t = i.split('$')

                    rule_triples_index.append((entity2id[h], relation2id[r], entity2id[t], p))
                    new_triple_only.append((entity2id[h], relation2id[r], entity2id[t]))

        return rule_triples_index, new_triple_only,rule_num


    new_triple, new_triple_only,rule_num = more_triple_p(args.new_triple_path)


    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding
    )
    
    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()
    
    if args.do_train:
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples,new_triple_only, nentity, nrelation, args.negative_sample_size, 'head-batch'),
            batch_size=args.batch_size,
            shuffle=True, 
            # num_workers=max(1, args.cpu_num//2),
            num_workers=1,
            collate_fn=TrainDataset.collate_fn
        )
        
        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, new_triple_only,nentity,  nrelation, args.negative_sample_size, 'tail-batch'),
            batch_size=args.batch_size,
            shuffle=True, 
            # num_workers=max(1, args.cpu_num//2),
            num_workers=1,
            collate_fn=TrainDataset.collate_fn
        )

        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)


        train_rule_triple = DataLoader(
            RuleTriple(new_triple, nentity, nrelation),
            batch_size=args.conclusion_batch_size,
            shuffle=True,
            # num_workers=max(1, args.cpu_num//2),
            num_workers=1,
            collate_fn=RuleTriple.collate_fn
        )

        train_rule_triple_iterator = Iterator(train_rule_triple)


        
        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()), 
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0
    
    step = init_step






    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))

    if args.use_rule:
        logging.info('rule_num = %d' % rule_num)
        logging.info('new_triple_num = %d' % len(new_triple_only))

    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
    
    # Set valid dataloader as it would be evaluated during training


    if args.do_train:
        logging.info('learning_rate = %f' % current_learning_rate)

        training_logs = []
        
        #Training Loop
        for step in range(init_step, args.max_steps):

            log = kge_model.train_step(kge_model, optimizer, train_iterator, train_rule_triple_iterator, args)
            
            training_logs.append(log)
            
            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()), 
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3
            
            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(kge_model, optimizer, save_variable_list, args)
                
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    if metric != "rule triple score:":
                        metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Training average', step, metrics)
                # training_logs = []

            if step % 1000 == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    if metric == "rule triple score:":
                        metrics[metric] = training_logs[-1][metric]
                log_metrics('Training average', step, metrics)
                training_logs = []

            if args.do_valid and step % args.valid_steps == 0 and step != 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics,_,__ = kge_model.test_step(kge_model, valid_triples, all_true_triples, args, id2entity, id2relation)
                log_metrics('Valid', step, metrics)

            if step % args.max_steps_per == 0:
                true_conclusions = conclusion_filtering(kge_model.entity_embedding.detach().cpu(),
                                               kge_model.relation_embedding.detach().cpu(),
                                               new_triple_only)
                train_triples = train_triples + true_conclusions
                train_dataloader_head = DataLoader(
                    TrainDataset(train_triples, new_triple_only, nentity, nrelation, args.negative_sample_size,
                                 'head-batch'),
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=max(1, args.cpu_num // 2),
                    collate_fn=TrainDataset.collate_fn
                )
                train_dataloader_tail = DataLoader(
                    TrainDataset(train_triples, new_triple_only, nentity, nrelation, args.negative_sample_size,
                                 'tail-batch'),
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=max(1, args.cpu_num // 2),
                    collate_fn=TrainDataset.collate_fn
                )
                train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
                out_map = defaultdict(list)
                in_map = defaultdict(list)
                entity = set()
                for e1, r, e2 in train_triples:
                    out_map[(e1, r)].append(e2)
                    in_map[(r, e2)].append(e1)
                    entity.add(e1)
                    entity.add(e2)
                entity = list(entity)
                conclusions = []
                one_time_reasoning(entity, rules, conclusions, in_map, out_map)
                train_rule_triple = DataLoader(
                    RuleTriple(conclusions, nentity, nrelation),
                    batch_size=args.conclusion_batch_size,
                    shuffle=True,
                    num_workers=max(1, args.cpu_num // 2),
                    collate_fn=RuleTriple.collate_fn
                )
                train_rule_triple_iterator = Iterator(train_rule_triple)
        
        save_variable_list = {
            'step': step, 
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, optimizer, save_variable_list, args)



    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics, head_triple, tail_triple = kge_model.test_step(kge_model, valid_triples, all_true_triples, args, id2entity, id2relation)
        log_metrics('Valid', step, metrics)

    if args.sparsity:
        logging.info('Evaluating on sparsity Dataset...')
        metrics, head_triple, tail_triple = kge_model.test_step(kge_model, sparse_triples, all_true_triples, args,id2entity, id2relation)
        log_metrics('sparsity', step, metrics)

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics, head_triple, tail_triple = kge_model.test_step(kge_model, test_triples, all_true_triples, args, id2entity, id2relation)
        log_metrics('Test', step, metrics)

    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics,head_triple, tail_triple = kge_model.test_step(kge_model, train_triples, all_true_triples, args, id2entity, id2relation)
        log_metrics('Test', step, metrics)



if __name__ == '__main__':
    main(parse_args())

#!/usr/bin/env python

import torch
import time
from data import train_data_prepare, train_data_prepare_augmented, test_data_prepare_augmented
from train import train
from test import test, test_data_prepare
from model import Net
from model_baseline import BaselineNet
import os
import pickle
import argparse
# os.environ["CUDA_VISIBLE_DEVICES"]="0"


def loadModel(word2num, num_classes, hyper, feat_list=[]):

    statement_word2num = word2num[0]
    subject_word2num = word2num[1]
    speaker_word2num = word2num[2]
    speaker_pos_word2num = word2num[3]
    state_word2num = word2num[4]
    party_word2num = word2num[5]
    context_word2num = word2num[6]
    justification_word2num = word2num[7]
    all_word2num = word2num[8]
    
    # Construct model instance
    print('  Constructing network model...')
    model = Net(
                len(all_word2num),
                num_classes,
                embed_dim = hyper['embed_dim'],
                statement_kernel_num = hyper['statement_kernel_num'],
                statement_kernel_size = hyper['statement_kernel_size'],

                subject_hidden_dim = hyper['subject_hidden_dim'],
                subject_lstm_nlayers = hyper['subject_lstm_nlayers'],
                subject_lstm_bidirectional = hyper['subject_lstm_bidirectional'],

                speaker_pos_hidden_dim = hyper['speaker_pos_hidden_dim'],
                speaker_pos_lstm_nlayers = hyper['speaker_pos_lstm_nlayers'],
                speaker_pos_lstm_bidirectional = hyper['speaker_pos_lstm_bidirectional'],

                context_hidden_dim = hyper['context_hidden_dim'],
                context_lstm_nlayers = hyper['context_lstm_nlayers'],
                context_lstm_bidirectional = hyper['context_lstm_bidirectional'],

                justification_hidden_dim = hyper['justification_hidden_dim'],
                justification_lstm_nlayers = hyper['justification_lstm_nlayers'],
                justification_lstm_bidirectional = hyper['justification_lstm_bidirectional'],

                dropout_query = hyper['dropout_query'],
                dropout_features = hyper['dropout_features'],

                augmented_feat = feat_list,
                )

    print("Hyperparams are:")
    for key in hyper:
        print(key, ": ", hyper[key])

    return model

def driver(args,train_file, valid_file, test_file, output_file, dataset, mode, features, pathModel, hyper, feat_list=[]):
    '''
    Arguments
    ----------
    train_file: path to the training file
    valid_file: path to the validation file
    test_file: path to the testing file
    output_file: path to the output predictions to be saved
    dataset: 'LIAR' or 'LIAR-PLUS'
    mode: 'train' or 'test'
    pathModel: path to model saved weights
    '''

    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime
    parentPath = args.modelDir # './models/'
    


    #---Hyperparams
    nnArchitecture = 'fake-net'
    lr = hyper['lr']
    epoch = hyper['epoch']
    use_cuda = True
    # use_cuda = False
    num_classes = hyper['num_classes']

    train_file = os.path.join(args.dataDir, train_file)
    valid_file = os.path.join(args.dataDir, valid_file)
    test_file = os.path.join(args.dataDir, test_file)

    assert num_classes in [2, 6]
    train_wikidict_file =  os.path.join(args.featDir, "train_wikictionary_dict.txt")
    test_wikidict_file = os.path.join(args.featDir, "test_wikictionary_dict.txt")
    valid_wikidict_file = os.path.join(args.featDir, "valid_wikictionary_dict.txt")

    train_liwc_file = os.path.join(args.featDir, "train_liwc_dict.txt")
    test_liwc_file = os.path.join(args.featDir, "test_liwc_dict.txt")
    valid_liwc_file = os.path.join(args.featDir, "valid_liwc_dict.txt")

    train_credit_file = "label_rel_counts"
    test_credit_file = "test_label_rel_counts" # valid and test can shares

    bert_dir = os.path.join(args.featDir,"BERT_feats")
    train_row_to_json = os.path.join(args.featDir, "train2.tsv_row_idx_tab_json_id.txt")
    test_row_to_json = os.path.join(args.featDir, "test2.tsv_row_idx_tab_json_id.txt")
    valid_row_to_json = os.path.join(args.featDir, "val2.tsv_row_idx_tab_json_id.txt")
    #-----------------TRAINING--------------
    if mode == 'train':
        #---prepare data
        if features == 'augmented':
                    
            train_samples, word2num = train_data_prepare_augmented(train_file, num_classes, dataset_name,
                        wikidict_filename= train_wikidict_file, liwcdict_filename=train_liwc_file, creditdict_filename = train_credit_file, bert_feat_dir=bert_dir, row_to_json= train_row_to_json)
            valid_samples = test_data_prepare_augmented(valid_file, word2num, 'valid', num_classes, dataset_name,
                        wikidict_filename= valid_wikidict_file, liwcdict_filename=valid_liwc_file, creditdict_filename = test_credit_file, bert_feat_dir=bert_dir, row_to_json= valid_row_to_json)
            # test_samples = test_data_prepare_augmented(test_file, word2num, 'test', num_classes, dataset_name,
            #             wikidict_filename= test_wikidict_file, liwcdict_filename=test_liwc_file, creditdict_filename = test_credit_file, bert_feat_dir=bert_dir, row_to_json= test_row_to_json)
            
        else:
            
            train_samples, word2num = train_data_prepare(train_file, num_classes, dataset_name)
            valid_samples = test_data_prepare(valid_file, word2num, 'valid', num_classes, dataset_name)
            # test_samples = test_data_prepare(test_file, word2num, 'test', num_classes, dataset_name)

        model = loadModel(word2num, num_classes, hyper, feat_list=feat_list)
        
        #---train and validate
        if features == 'augmented':
        	model, val_acc = train(train_samples, valid_samples, lr, epoch, model, num_classes, use_cuda, word2num,
                hyper, nnArchitecture, timestampLaunch, featuretype ='augmented', augment_feat=feat_list)
        else:
        	model, val_acc = train(train_samples, valid_samples, lr, epoch, model, num_classes, use_cuda, word2num, hyper, nnArchitecture, timestampLaunch)

        #---save model and embeddings
        pathModel = None



    #-----------------TESTING------------------

    if pathModel != None:
        pathModel = os.path.join(parentPath,pathModel) 

        modelCheckpoint = torch.load(pathModel, map_location=lambda storage, loc: storage)
        word2num = modelCheckpoint['word2num']
        hyper = modelCheckpoint['hyper']
        try:
            num_classes = hyper['num_classes']
        except:
            num_classes = 6

        if features == 'augmented':
            test_samples = test_data_prepare_augmented(test_file, word2num, 'test', num_classes, dataset_name,
                        wikidict_filename= test_wikidict_file, liwcdict_filename=test_liwc_file, creditdict_filename = test_credit_file, bert_feat_dir=bert_dir, row_to_json= test_row_to_json)
        else:
            test_samples = test_data_prepare(test_file, word2num, 'test', num_classes, dataset_name)

        model = loadModel(word2num, num_classes, hyper,feat_list=feat_list)
        
        device = torch.device('cuda') if use_cuda else torch.device('cpu')
        model.to(device)
        model.load_state_dict(modelCheckpoint['state_dict'])
        print("LOADED FROM PATHMODEL:", pathModel)

    else:
        print("PATHMODEL could not be loaded:", pathModel)

    test_acc = test(test_samples, output_file, model, num_classes, use_cuda, feat_list=feat_list)
        





#---HYPERPARAMETERS

hyper = {
'num_classes': 6,
'epoch': 30,
'lr': 0.001,
'embed_dim': 100,
'statement_kernel_num': 64,
'statement_kernel_size': [3, 4, 5],

'subject_hidden_dim': 8,
'subject_lstm_nlayers': 2,
'subject_lstm_bidirectional': True,

'speaker_pos_hidden_dim': 8,
'speaker_pos_lstm_nlayers': 2,
'speaker_pos_lstm_bidirectional': True,

'context_hidden_dim': 16,
'context_lstm_nlayers': 2,
'context_lstm_bidirectional': True,

'justification_hidden_dim': 32,
'justification_lstm_nlayers': 2,
'justification_lstm_bidirectional': True,

'dropout_query': 0.5,
'dropout_features': 0.7
}

dataset_name = 'LIAR-PLUS'


parser = argparse.ArgumentParser()
parser.add_argument('--mode', default="train", type=str,choices=['train','test'])
parser.add_argument('--feature_type', default="baseline", type=str,choices=['baseline','augmented'])
parser.add_argument('--modelDir', default='./models', type=str,help='model directory')
parser.add_argument('--pathModel', default=None, type=str,help='model name')
parser.add_argument('--feat_list', nargs='+')

parser.add_argument('--dataDir', default='./data/', type=str,help='directory for original data')
parser.add_argument('--featDir', default='./feats/', type=str,help='directory for extended features')

args = parser.parse_args()

mode = args.mode
features = args.feature_type
pathModel = args.pathModel

if args.feat_list is not None:
    feat_list = args.feat_list
else:
    feat_list = []


print("features:", features)
print("feat_list:",feat_list)

if mode == 'test':
    assert pathModel != None, "pathModel cannot be None if testing"


if dataset_name == 'LIAR':
    driver(args,'train.tsv', 'valid.tsv', 'test.tsv', 'predictions.txt', dataset_name, mode, features, pathModel, hyper, feat_list=feat_list)
else:
    driver(args,'train2.tsv', 'val2.tsv', 'test2.tsv', 'predictions.txt', dataset_name, mode, features, pathModel, hyper, feat_list=feat_list)

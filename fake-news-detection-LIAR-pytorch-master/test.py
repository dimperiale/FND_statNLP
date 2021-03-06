import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import DataSample, dataset_to_variable, test_data_prepare
import numpy as np
from sklearn.metrics import classification_report

num_to_label_6_way_classification = [
    'pants-fire',
    'false',
    'barely-true',
    'half-true',
    'mostly-true',
    'true'
]

num_to_label_2_way_classification = ['false', 'true']

def test(test_samples, test_output, model, classification_type, use_cuda = False, feat_list=[]):

    model.eval()
    featuretype = "augmented" if len(feat_list) >0 else "baseline"
    test_samples = dataset_to_variable(test_samples, use_cuda, featuretype = featuretype, augmented_feat = feat_list)
    out = open(test_output, 'w', buffering=1)
    acc = 0
    
    y_true = []
    y_pred = []

    for sample in test_samples:
        prediction = model(sample, augmented_feat=feat_list)
        prediction = int(np.argmax(prediction.cpu().data.numpy()))
        #---choose 6 way or binary classification 
        if classification_type == 2:
            out.write(num_to_label_2_way_classification[prediction]+'\n')
        else:
            out.write(num_to_label_6_way_classification[prediction]+'\n')

        if prediction == sample.label:
            acc += 1
        
        y_true.append(sample.label)
        y_pred.append(prediction)

    acc /= len(test_samples)
    print('  Test Accuracy: {:.3f}'.format(acc))
    out.close()

    target_names = num_to_label_6_way_classification 
    print(classification_report(y_true, y_pred, target_names=target_names,digits=5))

    return acc


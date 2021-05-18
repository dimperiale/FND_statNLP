# Fake News Detection

## Dataset
The [LIAR dataset](https://github.com/thiagorainmaker77/liar_dataset) consists of 12,836 short statements taken from POLITIFACT and labeled by humans for truthfulness, subject, context/venue, speaker, state, party, and prior history. For truthfulness, the LIAR dataset has six labels: pants-fire, false, mostly-false, half-true, mostly-true, and true. These six label sets are relatively balanced in size. The statements were collected from a variety of broadcasting mediums, like TV interviews, speeches, tweets, debates, and they cover a broad range of topics such as the economy, health care, taxes and election. The [LIAR-PLUS](https://github.com/Tariq60/LIAR-PLUS) dataset is an extension to the LIAR dataset by automatically extracting for each claim the justification that humans have provided in the fact-checking article associated with the claim.

## Architecture

## Network Architecture
![Screenshot 1](https://github.com/dimperiale/FND_statNLP/blob/main/fake-news-detection-LIAR-pytorch-master/fakenet_augmented.png "Net")



## How to Use

To train a model using all the extra features we created, execute the following.

python -u  main.py \
    --mode train --feature_type augmented \
    --feat_list  wiki_liwc_dict wiki_bert_fea

Change the mode to “test” and put the name of the saved model in the variable “pathModel”.

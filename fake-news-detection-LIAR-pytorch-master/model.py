import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math


class Net(nn.Module):

	def __init__(self,

				 vocab_dim,
				 num_classes,

				 embed_dim = 100,
				 statement_kernel_num = 14,
				 statement_kernel_size = [3, 4, 5],

				 subject_hidden_dim = 5,
				 subject_lstm_nlayers = 2,
				 subject_lstm_bidirectional = True,

				 speaker_pos_hidden_dim = 5,
				 speaker_pos_lstm_nlayers = 2,
				 speaker_pos_lstm_bidirectional = True,

				 context_hidden_dim = 6,
				 context_lstm_nlayers = 2,
				 context_lstm_bidirectional = True,

				 justification_hidden_dim = 6,
				 justification_lstm_nlayers = 2,
				 justification_lstm_bidirectional = True,

				 wiki_liwc_input_dim = 69, # 5 + 65
				 wiki_liwc_nn_dim = 10,

				 bert_feat_input_dim = 768*3, 
				 bert_feat_dense_1_dim = 1024,
				 bert_feat_dense_2_dim = 100,
				 bert_feat_dense_out_dim = 20,


				 credit_history_hidden_dim = 6,
				 credit_history_dense_dim = 10,

				 dropout_query = 0.5,
				 dropout_features = 0.5,
				 
				 augmented_feat = [], ):

		# Statement CNN
		super(Net, self).__init__()

		# import pdb; pdb.set_trace()
		self.num_classes = num_classes

		self.vocab_dim = vocab_dim
		self.embed_dim = embed_dim
		self.statement_kernel_num = statement_kernel_num
		self.statement_kernel_size = statement_kernel_size

		self.embedding = nn.Embedding(self.vocab_dim, self.embed_dim)
		# conv layer with in_channel = 1, out_channel = statement_kernel_num 
		# and kernel spatial size (rectangle) = (kernel_, embed_dim)
		self.statement_convs = nn.ModuleList()
		for kernel_ in self.statement_kernel_size:
			self.statement_convs.append(nn.Conv2d(1, self.statement_kernel_num, (kernel_, self.embed_dim)))

		# Subject
		self.subject_lstm_nlayers = subject_lstm_nlayers
		self.subject_lstm_num_direction = 2 if subject_lstm_bidirectional else 1
		self.subject_hidden_dim = subject_hidden_dim

		self.subject_lstm = nn.LSTM(
			input_size = self.embed_dim,
			hidden_size = self.subject_hidden_dim,
			num_layers = self.subject_lstm_nlayers,
			batch_first = True,
			bidirectional = subject_lstm_bidirectional
		)

		# Speaker
		
		# Speaker Position
		self.speaker_pos_lstm_nlayers = speaker_pos_lstm_nlayers
		self.speaker_pos_lstm_num_direction = 2 if speaker_pos_lstm_bidirectional else 1
		self.speaker_pos_hidden_dim = speaker_pos_hidden_dim

		self.speaker_pos_lstm = nn.LSTM(
			input_size = self.embed_dim,
			hidden_size = self.speaker_pos_hidden_dim,
			num_layers = self.speaker_pos_lstm_nlayers,
			batch_first = True,
			bidirectional = speaker_pos_lstm_bidirectional
		)

		# State
		
		# Party
		
		# Context
		self.context_lstm_nlayers = context_lstm_nlayers
		self.context_lstm_num_direction = 2 if context_lstm_bidirectional else 1
		self.context_hidden_dim = context_hidden_dim

		self.context_lstm = nn.LSTM(
			input_size = self.embed_dim,
			hidden_size = self.context_hidden_dim,
			num_layers = self.context_lstm_nlayers,
			batch_first = True,
			bidirectional = context_lstm_bidirectional
		)

		# Justification
		self.justification_lstm_nlayers = justification_lstm_nlayers
		self.justification_lstm_num_direction = 2 if justification_lstm_bidirectional else 1
		self.justification_hidden_dim = justification_hidden_dim

		self.justification_lstm = nn.LSTM(
			input_size = self.embed_dim,
			hidden_size = self.justification_hidden_dim,
			num_layers = self.justification_lstm_nlayers,
			batch_first = True,
			bidirectional = justification_lstm_bidirectional
		)


		self.dropout_query = nn.Dropout(dropout_query)
		self.dropout_features = nn.Dropout(dropout_features)
		self.query_dim = self.subject_lstm_nlayers * self.subject_lstm_num_direction \
						+ self.embed_dim \
						+ self.speaker_pos_lstm_nlayers * self.speaker_pos_lstm_num_direction \
						+ self.embed_dim \
						+ self.embed_dim \
						+ self.context_lstm_nlayers * self.context_lstm_num_direction \
						+ self.justification_lstm_nlayers * self.justification_lstm_num_direction

		# additional augment features
		# WIKI LIWC DICT
		if 'wiki_liwc_dict' in augmented_feat:
			self.wiki_liwc_input_dense = nn.Linear(wiki_liwc_input_dim,wiki_liwc_nn_dim)
			self.query_dim += wiki_liwc_nn_dim;
		# wiki top sentences bert features
		if 'wiki_bert_feat' in augmented_feat:
			self.wiki_bert_input_dense = nn.Linear(bert_feat_input_dim, bert_feat_dense_1_dim)
			self.wiki_bert_dense1_dense = nn.Linear(bert_feat_dense_1_dim,bert_feat_dense_2_dim)
			self.wiki_bert_output_dense = nn.Linear(bert_feat_dense_2_dim, bert_feat_dense_out_dim)

			self.query_dim += bert_feat_dense_out_dim;
			
		if "credit_history_feat" in augmented_feat:
			self.credit_history_dense_nlayers = nn.Linear(credit_history_hidden_dim, credit_history_dense_dim)

			self.query_dim += credit_history_dense_dim


		self.fc_query = nn.Linear(self.query_dim, self.embed_dim)
		self.fc_att = nn.Linear(self.embed_dim, self.embed_dim)
		self.fc_conv = nn.Linear(self.embed_dim, self.embed_dim)
		self.fc_cat = nn.Linear(self.embed_dim, self.embed_dim)
		self.fc = nn.Linear(len(self.statement_kernel_size) * self.statement_kernel_num + self.query_dim,
							self.num_classes)

	def forward(self, sample, augmented_feat=[]):

		
		statement = Variable(sample.statement).unsqueeze(0)
		subject = Variable(sample.subject).unsqueeze(0)
		speaker = Variable(sample.speaker).unsqueeze(0)
		speaker_pos = Variable(sample.speaker_pos).unsqueeze(0)
		state = Variable(sample.state).unsqueeze(0)
		party = Variable(sample.party).unsqueeze(0)
		context = Variable(sample.context).unsqueeze(0)
		justification = Variable(sample.justification).unsqueeze(0)

		batch = 1 # Current support one sample per time
				  # TODO: Increase batch number


		# Subject
		subject_ = self.embedding(subject) # 1*W*D
		_, (subject_, _) = self.subject_lstm(subject_) # (layer x dir) * batch * hidden
		subject_ = F.max_pool1d(subject_, self.subject_hidden_dim).view(1, -1) # (layer x dir) * batch * 1 -> 1*(layer x dir)

		# Speaker
		speaker_ = self.embedding(speaker).squeeze(0) # 1*1*D -> 1*D

		# Speaker Position
		speaker_pos_ = self.embedding(speaker_pos)
		_, (speaker_pos_, _) = self.speaker_pos_lstm(speaker_pos_)
		speaker_pos_ = F.max_pool1d(speaker_pos_, self.speaker_pos_hidden_dim).view(1, -1)

		# State
		state_ = self.embedding(state).squeeze(0)

		# Party
		party_ = self.embedding(party).squeeze(0)

		# Context
		context_ = self.embedding(context)
		_, (context_, _) = self.context_lstm(context_)
		context_ = F.max_pool1d(context_, self.context_hidden_dim).view(1, -1)

		# Justification
		justification_ = self.embedding(justification)
		_, (justification_, _) = self.justification_lstm(justification_)
		justification_ = F.max_pool1d(justification_, self.justification_hidden_dim).view(1, -1)
		
		
		if 'wiki_liwc_dict' in augmented_feat:
			dictionaries_vect = Variable(sample.dictionaries_vect).unsqueeze(0)
			wiki_liwc_feats_ = self.wiki_liwc_input_dense(dictionaries_vect)
		if 'wiki_bert_feat' in augmented_feat:
			if sample.wiki_top_bert_feat.shape[0]==768:
				sample.wiki_top_bert_feat = torch.cat([sample.wiki_top_bert_feat,sample.wiki_top_bert_feat,sample.wiki_top_bert_feat],0)
			elif sample.wiki_top_bert_feat.shape[0]==768*2:
				sample.wiki_top_bert_feat = torch.cat([sample.wiki_top_bert_feat,sample.wiki_top_bert_feat[:768]],0)

			wiki_top_bert_feat = Variable(sample.wiki_top_bert_feat).unsqueeze(0)
			wiki_top_bert_feat_ = self.wiki_bert_input_dense(wiki_top_bert_feat)
			wiki_top_bert_feat_ = self.wiki_bert_dense1_dense(wiki_top_bert_feat_)
			wiki_top_bert_feat_ = self.wiki_bert_output_dense(wiki_top_bert_feat_)

		if "credit_history_feat" in augmented_feat:
			credit_history_ = Variable(sample.credithistory_vect).unsqueeze(0)
			credit_history_ = self.credit_history_dense_nlayers(credit_history_ )


		# Statement
		query_element_tuple = (subject_, speaker_, speaker_pos_, state_, party_, context_, justification_)
		if 'wiki_liwc_dict' in augmented_feat:
			query_element_tuple += (wiki_liwc_feats_,)
		if 'wiki_bert_feat' in augmented_feat:
			query_element_tuple += (wiki_top_bert_feat_,)
		if "credit_history_feat" in augmented_feat:
			query_element_tuple += (credit_history_ ,)

		query = torch.cat(query_element_tuple, 1)
		query = F.leaky_relu(self.fc_query(query))
		query = self.dropout_query(query)
		query_att = self.fc_att(query).view(1, -1)

		# Contextualize the conv filters
		query_conv = self.fc_conv(query).view(1, -1)
		for conv in self.statement_convs:
			conv.weight = nn.Parameter(conv.weight * query_conv)



		statement_ = self.embedding(statement).unsqueeze(0) # 1*W*D -> 1*1*W*D
		# Attention
		# import pdb; pdb.set_trace()
		alpha = (query_att * statement_).sum(dim=-1).view(-1, 1)
		alpha = F.softmax(alpha / math.sqrt(self.embed_dim), dim=0)
		statement_ = alpha * statement_
		statement_ = [F.relu(conv(statement_)).squeeze(3) for conv in self.statement_convs] # 1*1*W*1 -> 1*Conv-filters*(W-1) x len(convs)
		statement_ = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in statement_] # 1*Conv-filters*1 -> 1*Conv-filters x len(convs)
		statement_ = torch.cat(statement_, 1)  # 1*len(convs)

		# Concatenate
		feats_element_tuple = (statement_, subject_, speaker_, speaker_pos_, state_, party_, context_, justification_)
		if 'wiki_liwc_dict' in augmented_feat:
			feats_element_tuple += (wiki_liwc_feats_,)
		if 'wiki_bert_feat' in augmented_feat:
			feats_element_tuple += (wiki_top_bert_feat_,)
		if "credit_history_feat" in augmented_feat:
			feats_element_tuple += (credit_history_,)
		features = torch.cat(feats_element_tuple, 1)
		features = self.dropout_features(features)
		out = self.fc(features)
		out = F.log_softmax(out, dim=-1)

		return out

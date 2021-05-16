import re
from numpy import not_equal
import torch
import pickle
import os

liwc_cats = ['Funct', 'Pronoun', 'Ppron', 'I', 'We', 'You', 'SheHe', 'They', 'Ipron', 
             'Article', 'Verbs', 'AuxVb', 'Past', 'Present', 'Future', 'Adverbs', 'Prep', 
             'Conj', 'Negate', 'Quant', 'Numbers', 'Swear', 'Social', 'Family', 'Friends', 'Humans', 
             'Affect', 'Posemo', 'Negemo', 'Anx', 'Anger', 'Sad', 'CogMech', 'Insight', 'Cause', 'Discrep', 
             'Tentat', 'Certain', 'Inhib', 'Incl', 'Excl', 'Percept', 'See', 'Hear', 'Feel', 'Bio', 'Body', 
             'Health', 'Sexual', 'Ingest', 'Relativ', 'Motion', 'Space', 'Time', 'Work', 'Achiev', 'Leisure', 
             'Home', 'Money', 'Relig', 'Death', 'Assent', 'Nonflu', 'Filler']



wiki_cats = ["act_adverbs", "comparative_forms", "manner_adverbs", "modal_adverbs", "superlative_forms"]


label_to_number_6_way_classification = {
	'pants-fire': 0,
	'false': 1,
	'barely-true': 2,
	'half-true': 3,
	'mostly-true': 4,
	'true': 5
}

label_to_number_2_way_classification = {
	'pants-fire': 0,
	'false': 0,
	'barely-true': 0,
	'half-true': 1,
	'mostly-true': 1,
	'true': 1
}

num_to_label_6_way_classification = [
	'pants-fire',
	'false',
	'barely-true',
	'half-true',
	'mostly-true',
	'true'
]

def dataset_to_variable(dataset, use_cuda, featuretype, augmented_feat=[]):


	for i in range(len(dataset)):

		dataset[i].statement = torch.LongTensor(dataset[i].statement)
		dataset[i].subject = torch.LongTensor(dataset[i].subject)
		dataset[i].speaker = torch.LongTensor([dataset[i].speaker])
		dataset[i].speaker_pos = torch.LongTensor(dataset[i].speaker_pos)
		dataset[i].state = torch.LongTensor([dataset[i].state])
		dataset[i].party = torch.LongTensor([dataset[i].party])
		dataset[i].context = torch.LongTensor(dataset[i].context)
		dataset[i].justification = torch.LongTensor(dataset[i].justification)

		if featuretype == "augmented":
			if 'wiki_liwc_dict' in augmented_feat:
				dataset[i].dictionaries_vect = torch.FloatTensor(dataset[i].dictionaries_vect) # len  70
			if 'credit_history_feat' in augmented_feat:
				dataset[i].credithistory_vect = torch.FloatTensor(dataset[i].credithistory_vect) # len 6
			if 'wiki_bert_feat' in augmented_feat:
				dataset[i].wiki_top_bert_feat = torch.FloatTensor(dataset[i].wiki_top_bert_feat)




		if use_cuda:
			dataset[i].statement = dataset[i].statement.cuda()
			dataset[i].subject = dataset[i].subject.cuda()
			dataset[i].speaker = dataset[i].speaker.cuda()
			dataset[i].speaker_pos = dataset[i].speaker_pos.cuda()
			dataset[i].state = dataset[i].state.cuda()
			dataset[i].party = dataset[i].party.cuda()
			dataset[i].context = dataset[i].context.cuda()
			dataset[i].justification = dataset[i].justification.cuda()

			if featuretype == "augmented":
				if 'wiki_liwc_dict' in augmented_feat:
					dataset[i].dictionaries_vect = dataset[i].dictionaries_vect.cuda()
				if 'credit_history_feat' in augmented_feat:
					dataset[i].credithistory_vect = dataset[i].credithistory_vect.cuda()
				if 'wiki_bert_feat' in augmented_feat:
					dataset[i].wiki_top_bert_feat = dataset[i].wiki_top_bert_feat.cuda() # or loading when training, if mem not enough


	return dataset

class DataSample:
	def __init__(self,
		label,
		statement,
		subject,
		speaker,
		speaker_pos,
		state,
		party,
		context,
		justification,
		num_classes,
		dataset_name):

		#---choose 6 way or binary classification 
		if num_classes == 2:
			self.label = label_to_number_2_way_classification.get(label, -1)
		else:
			self.label = label_to_number_6_way_classification.get(label, -1)


		self.statement = re.sub('[().]', '', statement).strip().split()
		while len(self.statement) < 5:
			self.statement.append('<no>')
		self.subject = subject.strip().split(',')
		self.speaker = speaker
		self.speaker_pos = speaker_pos.strip().split()
		self.state = state
		self.party = party
		self.context = context.strip().split()
		self.justification = re.sub('[().]', '', justification).strip().split()
		
		if len(self.statement) == 0:
			self.statement = ['<no>']
		if len(self.subject) == 0:
			self.subject = ['<no>']
		if len(self.speaker) == 0:
			self.speaker = '<no>'
		if len(self.speaker_pos) == 0:
			self.speaker_pos = ['<no>']
		if len(self.state) == 0:
			self.state = '<no>'
		if len(self.party) == 0:
			self.party = '<no>'
		if len(self.context) == 0:
			self.context = ['<no>']
		if len(self.justification) == 0:
			self.justification = ['<no>']
            
            
class DataSample_augmented:
	def __init__(self,
		label,
		statement,
		subject,
		speaker,
		speaker_pos,
		state,
		party,
		context,
		justification,
		num_classes,
		dataset_name,
		wikictionary=None,
		liwc=None,
		credit=None,
		bert_feat=None,
		):

		#---choose 6 way or binary classification 
		if num_classes == 2:
			self.label = label_to_number_2_way_classification.get(label, -1)
		else:
			self.label = label_to_number_6_way_classification.get(label, -1)


		self.statement = re.sub('[().]', '', statement).strip().split()
		while len(self.statement) < 5:
			self.statement.append('<no>')
		self.subject = subject.strip().split(',')
		self.speaker = speaker
		self.speaker_pos = speaker_pos.strip().split()
		self.state = state
		self.party = party
		self.context = context.strip().split()
		self.justification = re.sub('[().]', '', justification).strip().split()
		
		if len(self.statement) == 0:
			self.statement = ['<no>']
		if len(self.subject) == 0:
			self.subject = ['<no>']
		if len(self.speaker) == 0:
			self.speaker = '<no>'
		if len(self.speaker_pos) == 0:
			self.speaker_pos = ['<no>']
		if len(self.state) == 0:
			self.state = '<no>'
		if len(self.party) == 0:
			self.party = '<no>'
		if len(self.context) == 0:
			self.context = ['<no>']
		if len(self.justification) == 0:
			self.justification = ['<no>']
            
        # wiki dictionary & liwc dictionary
		if wikictionary is not None and liwc is not None:
			wiki_vec = []
			liwc_vec = []

			for i in range(len(wiki_cats)):
				wiki_cat = wiki_cats[i]
				wiki_vec.append(wikictionary.get(wiki_cat, 0))

			for i in range(len(liwc_cats)):
				liwc_cat = liwc_cats[i]
				liwc_vec.append(liwc.get(liwc_cat, 0))


			self.dictionaries_vect = wiki_vec + liwc_vec

		if credit is not None:
			credit_vec = []

			for i in range(len(num_to_label_6_way_classification)):
				credit_cat = num_to_label_6_way_classification[i]
				credit_vec.append(credit.get(credit_cat, 0))

			self.credithistory_vect = credit_vec
				
		if bert_feat is not None:
			self.wiki_top_bert_feat = bert_feat.view(-1)

            

#---Train data prep
def count_in_vocab(dict, word):
	if word not in dict:
		dict[word] = len(dict)
		return dict[word]
	else:
		return dict[word]

def train_data_prepare(train_filename, num_classes, dataset_name):
	print("Preparing data from: " + train_filename)
	train_file = open(train_filename, 'rb')

	lines = train_file.read()
	lines = lines.decode("utf-8")
	train_samples = []
	statement_word2num = {'<unk>' : 0}
	subject_word2num = {'<unk>' : 0}
	speaker_word2num = {'<unk>' : 0}
	speaker_pos_word2num = {'<unk>' : 0}
	state_word2num = {'<unk>' : 0}
	party_word2num = {'<unk>' : 0}
	context_word2num = {'<unk>' : 0}
	justification_word2num = {'<unk>' : 0}
	all_word2num = {'<unk>' : 0}

	fault=0
	try:
		for line in lines.strip().split('\n'):
			tmp = line.strip().split('\t')
			# tmp = list(filter(None, tmp))
			# if len(tmp)!=16:
			# 	fault +=1
			# 	print("len(tmp):", len(tmp), " ", tmp )
				# import pdb; pdb.set_trace()
			# import pdb; pdb.set_trace()
			# if tmp[2] not in num_to_label_6_way_classification:
			# 	# import pdb; pdb.set_trace()
			# 	fault +=1
			# 	print("len(tmp):", len(tmp), " ", tmp )
			# import pdb; pdb.set_trace()

			if dataset_name == 'LIAR':
				#---LIAR
				while len(tmp) < 14:
					tmp.append('')
				p = DataSample(tmp[1], tmp[2], tmp[3], tmp[4], tmp[5] , tmp[6], tmp[7], tmp[13], '', num_classes, dataset_name)
			else:
				#---LIAR-PLUS
				while len(tmp) < 16:
					tmp.append('')
				if tmp[2] not in num_to_label_6_way_classification:
					p = DataSample(tmp[1], tmp[2], tmp[3], tmp[4], tmp[5] , tmp[6], tmp[7], tmp[13], tmp[14], num_classes, dataset_name)
				else:
					p = DataSample(tmp[2], tmp[3], tmp[4], tmp[5], tmp[6] , tmp[7], tmp[8], tmp[14], tmp[15], num_classes, dataset_name)

			# for i in range(len(p.statement)):
			# 	p.statement[i] = count_in_vocab(statement_word2num, p.statement[i])
			# for i in range(len(p.subject)):
			# 	p.subject[i] = count_in_vocab(subject_word2num, p.subject[i])
			# p.speaker = count_in_vocab(speaker_word2num, p.speaker)
			# for i in range(len(p.speaker_pos)):
			# 	p.speaker_pos[i] = count_in_vocab(speaker_pos_word2num, p.speaker_pos[i])
			# p.state = count_in_vocab(state_word2num, p.state)
			# p.party = count_in_vocab(party_word2num, p.party)
			# for i in range(len(p.context)):
			# 	p.context[i] = count_in_vocab(context_word2num, p.context[i])
			# for i in range(len(p.justification)):
			# 	p.justification[i] = count_in_vocab(justification_word2num, p.justification[i])


			for i in range(len(p.statement)):
				p.statement[i] = count_in_vocab(all_word2num, p.statement[i])
			for i in range(len(p.subject)):
				p.subject[i] = count_in_vocab(all_word2num, p.subject[i])
			p.speaker = count_in_vocab(all_word2num, p.speaker)
			for i in range(len(p.speaker_pos)):
				p.speaker_pos[i] = count_in_vocab(all_word2num, p.speaker_pos[i])
			p.state = count_in_vocab(all_word2num, p.state)
			p.party = count_in_vocab(all_word2num, p.party)
			for i in range(len(p.context)):
				p.context[i] = count_in_vocab(all_word2num, p.context[i])
			for i in range(len(p.justification)):
				p.justification[i] = count_in_vocab(all_word2num, p.justification[i])


			
			train_samples.append(p)
	except:
		print("except")
		import pdb; pdb.set_trace()

	print("fault:", fault)

	word2num = [statement_word2num,
				subject_word2num,
				speaker_word2num,
				speaker_pos_word2num,
				state_word2num,
				party_word2num,
				context_word2num,
				justification_word2num,
				all_word2num]

	print("  "+str(len(train_samples))+" samples")

	print("  Statement Vocabulary Size: " + str(len(statement_word2num)))
	print("  Subject Vocabulary Size: " + str(len(subject_word2num)))
	print("  Speaker Vocabulary Size: " + str(len(speaker_word2num)))
	print("  Speaker Position Vocabulary Size: " + str(len(speaker_pos_word2num)))
	print("  State Vocabulary Size: " + str(len(state_word2num)))
	print("  Party Vocabulary Size: " + str(len(party_word2num)))
	print("  Context Vocabulary Size: " + str(len(context_word2num)))
	print("  Justification Vocabulary Size: " + str(len(justification_word2num)))
	print("  Vocabulary Size: " + str(len(all_word2num)))

	return train_samples, word2num



def train_data_prepare_augmented(train_filename, num_classes, dataset_name, wikidict_filename=None, liwcdict_filename=None, creditdict_filename = None, bert_feat_dir=None, row_to_json= None):
	print("Preparing data from: " + train_filename)
    
    
	train_file = open(train_filename, 'rb')
	if wikidict_filename is not None :
		with open(wikidict_filename, 'rb') as f:
			wikidict = pickle.load(f)
	if liwcdict_filename is not None:
		with open(liwcdict_filename, 'rb') as f:
			liwcdict = pickle.load(f)
    if credit_filename is not None:
		with open(credit_filename, 'rb') as f:
			creditdict = pickle.load(f)
	if row_to_json is not None:
		row_idx_to_jsonid = {}
		with open(row_to_json,'r') as f:
			for line in f:
				idx, json_id = line.strip().split('\t')
				row_idx_to_jsonid[int(idx)] = json_id

	lines = train_file.read()
	lines = lines.decode("utf-8")
	train_samples = []
	statement_word2num = {'<unk>' : 0}
	subject_word2num = {'<unk>' : 0}
	speaker_word2num = {'<unk>' : 0}
	speaker_pos_word2num = {'<unk>' : 0}
	state_word2num = {'<unk>' : 0}
	party_word2num = {'<unk>' : 0}
	context_word2num = {'<unk>' : 0}
	justification_word2num = {'<unk>' : 0}
	all_word2num = {'<unk>' : 0}

	fault=0
	try:
		row_idx = 0
		for line in lines.strip().split('\n'):
			tmp = line.strip().split('\t')

			wiki_dict_feat = None if wikidict_filename is None else wikidict[row_idx][1]
			liwc_dict_feat = None if liwcdict_filename is None else liwcdict[row_idx][1]
			credit_dict_feat = None if credit_filename is None else creditdict[row_idx][1]
			bert_feat = None if row_to_json==None else torch.load(os.path.join(bert_feat_dir,f"{train_filename}_{row_idx_to_jsonid[row_idx]}_top_3_wiki_bert_feats.pt"))

			if dataset_name == 'LIAR':
				#---LIAR
				while len(tmp) < 14:
					tmp.append('')
				p = DataSample_augmented(tmp[1], tmp[2], tmp[3], tmp[4], tmp[5] , tmp[6], tmp[7], tmp[13],'',\
                                         num_classes, dataset_name,
										 wikictionary=wiki_dict_feat,liwc=liwc_dict_feat,credit=credit_dict_feat,bert_feat=bert_feat, credit =)
			else:
				#---LIAR-PLUS
				while len(tmp) < 16:
					tmp.append('')
				if tmp[2] not in num_to_label_6_way_classification:
					p = DataSample_augmented(tmp[1], tmp[2], tmp[3], tmp[4], tmp[5] , tmp[6], tmp[7], tmp[13], tmp[14],\
                                	num_classes, dataset_name,
								   	wikictionary=wiki_dict_feat,liwc=liwc_dict_feat,credit=credit_dict_feat,bert_feat=bert_feat)
				else:
					p = DataSample_augmented(tmp[2], tmp[3], tmp[4], tmp[5], tmp[6] , tmp[7], tmp[8], tmp[14], tmp[15],\
                                   num_classes, dataset_name,
								   wikictionary=wiki_dict_feat,liwc=liwc_dict_feat,credit=credit_dict_feat,bert_feat=bert_feat)


			for i in range(len(p.statement)):
				p.statement[i] = count_in_vocab(all_word2num, p.statement[i])
			for i in range(len(p.subject)):
				p.subject[i] = count_in_vocab(all_word2num, p.subject[i])
			p.speaker = count_in_vocab(all_word2num, p.speaker)
			for i in range(len(p.speaker_pos)):
				p.speaker_pos[i] = count_in_vocab(all_word2num, p.speaker_pos[i])
			p.state = count_in_vocab(all_word2num, p.state)
			p.party = count_in_vocab(all_word2num, p.party)
			for i in range(len(p.context)):
				p.context[i] = count_in_vocab(all_word2num, p.context[i])
			for i in range(len(p.justification)):
				p.justification[i] = count_in_vocab(all_word2num, p.justification[i])


			row_idx+=1
			train_samples.append(p)
	except:
		print("except")
		import pdb; pdb.set_trace()

	print("fault:", fault)

	word2num = [statement_word2num,
				subject_word2num,
				speaker_word2num,
				speaker_pos_word2num,
				state_word2num,
				party_word2num,
				context_word2num,
				justification_word2num,
				all_word2num]

	print("  "+str(len(train_samples))+" samples")

	print("  Statement Vocabulary Size: " + str(len(statement_word2num)))
	print("  Subject Vocabulary Size: " + str(len(subject_word2num)))
	print("  Speaker Vocabulary Size: " + str(len(speaker_word2num)))
	print("  Speaker Position Vocabulary Size: " + str(len(speaker_pos_word2num)))
	print("  State Vocabulary Size: " + str(len(state_word2num)))
	print("  Party Vocabulary Size: " + str(len(party_word2num)))
	print("  Context Vocabulary Size: " + str(len(context_word2num)))
	print("  Justification Vocabulary Size: " + str(len(justification_word2num)))
	print("  Vocabulary Size: " + str(len(all_word2num)))

	return train_samples, word2num



#---Test data prep
def find_word(word2num, token):
	if token in word2num:
		return word2num[token]
	else:
		return word2num['<unk>']

def test_data_prepare(test_file, word2num, phase, num_classes, dataset_name):
	test_input = open(test_file, 'rb')
	test_data = test_input.read().decode('utf-8')
	test_input.close()

	statement_word2num = word2num[0]
	subject_word2num = word2num[1]
	speaker_word2num = word2num[2]
	speaker_pos_word2num = word2num[3]
	state_word2num = word2num[4]
	party_word2num = word2num[5]
	context_word2num = word2num[6]
	justification_word2num = word2num[7]
	all_word2num = word2num[8]

	test_samples = []

	fault=0
	for line in test_data.strip().split('\n'):
		tmp = line.strip().split('\t')
		
		if dataset_name == 'LIAR':
			while len(tmp) < 15:
				tmp.append('')
			p = DataSample(tmp[1], tmp[2], tmp[3], tmp[4], tmp[5] , tmp[6], tmp[7], tmp[13], '', num_classes, dataset_name)
		else:
			while len(tmp) < 16:
				tmp.append('')
			if tmp[2] not in num_to_label_6_way_classification:
				p = DataSample(tmp[1], tmp[2], tmp[3], tmp[4], tmp[5] , tmp[6], tmp[7], tmp[13], tmp[14], num_classes, dataset_name)
			else:
				p = DataSample(tmp[2], tmp[3], tmp[4], tmp[5], tmp[6] , tmp[7], tmp[8], tmp[14], tmp[15], num_classes, dataset_name)

		# for i in range(len(p.statement)):
		# 	p.statement[i] = find_word(statement_word2num, p.statement[i])
		# for i in range(len(p.subject)):
		# 	p.subject[i] = find_word(subject_word2num, p.subject[i])
		# p.speaker = find_word(speaker_word2num, p.speaker)
		# for i in range(len(p.speaker_pos)):
		# 	p.speaker_pos[i] = find_word(speaker_pos_word2num, p.speaker_pos[i])
		# p.state = find_word(state_word2num, p.state)
		# p.party = find_word(party_word2num, p.party)
		# for i in range(len(p.context)):
		# 	p.context[i] = find_word(context_word2num, p.context[i])
		# for i in range(len(p.justification)):
		# 	p.justification[i] = find_word(justification_word2num, p.justification[i])


		for i in range(len(p.statement)):
			p.statement[i] = find_word(all_word2num, p.statement[i])
		for i in range(len(p.subject)):
			p.subject[i] = find_word(all_word2num, p.subject[i])
		p.speaker = find_word(all_word2num, p.speaker)
		for i in range(len(p.speaker_pos)):
			p.speaker_pos[i] = find_word(all_word2num, p.speaker_pos[i])
		p.state = find_word(all_word2num, p.state)
		p.party = find_word(all_word2num, p.party)
		for i in range(len(p.context)):
			p.context[i] = find_word(all_word2num, p.context[i])
		for i in range(len(p.justification)):
			p.justification[i] = find_word(all_word2num, p.justification[i])


		test_samples.append(p)

	print("fault:", fault)

	return test_samples



def test_data_prepare_augmented(test_file, word2num, phase, num_classes, dataset_name, wikidict_filename=None, liwcdict_filename=None, creditdict_filename=None, bert_feat_dir=None, row_to_json= None):
	test_input = open(test_file, 'rb')
	test_data = test_input.read().decode('utf-8')
	test_input.close()
    
	if wikidict_filename is not None :
		with open(wikidict_filename, 'rb') as f:
			wikidict = pickle.load(f)
	if liwcdict_filename is not None:
		with open(liwcdict_filename, 'rb') as f:
			liwcdict = pickle.load(f)
	if credit_filename is not None:
		with open(credit_filename, 'rb') as f:
			creditdict = pickle.load(f)
	if row_to_json is not None:
		row_idx_to_jsonid = {}
		with open(row_to_json,'r') as f:
			for line in f:
				idx, json_id = line.strip().split('\t')
				row_idx_to_jsonid[int(idx)] = json_id

	statement_word2num = word2num[0]
	subject_word2num = word2num[1]
	speaker_word2num = word2num[2]
	speaker_pos_word2num = word2num[3]
	state_word2num = word2num[4]
	party_word2num = word2num[5]
	context_word2num = word2num[6]
	justification_word2num = word2num[7]
	all_word2num = word2num[8]

	test_samples = []

	fault=0
	row_idx = 0
	for line in test_data.strip().split('\n'):
		tmp = line.strip().split('\t')

		wiki_dict_feat = None if wikidict_filename is None else wikidict[row_idx][1]
		liwc_dict_feat = None if liwcdict_filename is None else liwcdict[row_idx][1]
		credit_dict_feat = None if creditdict_filename is None else creditdict
		bert_feat = None if row_to_json==None else torch.load(os.path.join(bert_feat_dir,f"{test_file}_{row_idx_to_jsonid[row_idx]}_top_3_wiki_bert_feats.pt"))

		if dataset_name == 'LIAR':
			while len(tmp) < 15:
				tmp.append('')
			speaker = tmp[4]
			p = DataSample_augmented(tmp[1], tmp[2], tmp[3], tmp[4], tmp[5] , tmp[6], tmp[7], tmp[13], '', \
                                     num_classes, dataset_name,
									 wikictionary=wiki_dict_feat,liwc=liwc_dict_feat,credit = credit_dict_feat[speaker],bert_feat=bert_feat)
		else:
			while len(tmp) < 16:
				tmp.append('')
			if tmp[2] not in num_to_label_6_way_classification:
				speaker = tmp[4]
				p = DataSample_augmented(tmp[1], tmp[2], tmp[3], tmp[4], tmp[5] , tmp[6], tmp[7], tmp[13], tmp[14], \
                                         num_classes, dataset_name,
										 wikictionary=wiki_dict_feat,liwc=liwc_dict_feat,credit = credit_dict_feat[speaker],bert_feat=bert_feat)
			else:
				speaker = tmp[5]
				p = DataSample_augmented(tmp[2], tmp[3], tmp[4], tmp[5], tmp[6] , tmp[7], tmp[8], tmp[14], tmp[15], \
                                         num_classes, dataset_name,
										 wikictionary=wiki_dict_feat,liwc=liwc_dict_feat,credit = credit_dict_feat[speaker],bert_feat=bert_feat)



		for i in range(len(p.statement)):
			p.statement[i] = find_word(all_word2num, p.statement[i])
		for i in range(len(p.subject)):
			p.subject[i] = find_word(all_word2num, p.subject[i])
		p.speaker = find_word(all_word2num, p.speaker)
		for i in range(len(p.speaker_pos)):
			p.speaker_pos[i] = find_word(all_word2num, p.speaker_pos[i])
		p.state = find_word(all_word2num, p.state)
		p.party = find_word(all_word2num, p.party)
		for i in range(len(p.context)):
			p.context[i] = find_word(all_word2num, p.context[i])
		for i in range(len(p.justification)):
			p.justification[i] = find_word(all_word2num, p.justification[i])


		row_idx+=1
		test_samples.append(p)

	print("fault:", fault)

	return test_samples
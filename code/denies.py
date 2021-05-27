#!/usr/bin/python

import random
from optparse import OptionParser

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from utils import *
from models import *

######################################################################
		   			  #-- train and predict --#
######################################################################

def train_and_predict(train_loader, val_loader, test_loader, model, data_index, 
				num_epochs, learning_rate, patience, metrics):

	loss_fn = torch.nn.BCELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
	early_stopping = EarlyStopping(patience = patience, metrics = metrics, verbose = False)

	all_preds = []

	for epoch in range(num_epochs):
		train_loss = 0
		total_correct = 0
		model.train()
		for i, data in enumerate(train_loader, 0):
			optimizer.zero_grad()
			
			selected_data = (data[index] for index in data_index)
			pred_label = torch.squeeze(model(*selected_data).float())
			#data[-1] represents the labels
			loss = loss_fn(pred_label, data[-1])
			loss.backward()
			optimizer.step()


			train_loss += loss.item() * data[-1].size(0)

			correct = ((pred_label > 0.5) == data[-1]).sum().item()
			total_correct += correct

		train_loss = train_loss / len(train_loader.dataset)
		train_accuracy = float(total_correct) / len(train_loader.dataset)
		print('epoch: {} \ttraining loss: {:.6f} \ttraining acc: {:.6f}'.format(epoch + 1, train_loss, train_accuracy))

		val_loss = 0
		total_correct = 0

		model.eval()
		with torch.no_grad():
			for i, data in enumerate(val_loader, 0):

				selected_data = (data[index] for index in data_index)
				pred_label = torch.squeeze(model(*selected_data).float())

				acc, mcc = compute_metrics(data[-1], pred_label.numpy(), False)

				loss = loss_fn(pred_label, data[-1])
				val_loss += loss.item() * data[-1].size(0)

				correct = ((pred_label > 0.5) == data[-1]).sum().item()
				total_correct += correct

			val_loss = val_loss / len(val_loader.dataset)
			val_accuracy = float(total_correct) / len(val_loader.dataset)

			print('epoch: {} \tval loss: {:.6f} \tval acc: {:.6f} \tMCC: {:.6f}'.format(epoch + 1, val_loss, val_accuracy, mcc))

		if metrics == 'val_loss':
			early_stopping(val_loss, model)
		else:
			early_stopping(mcc, model)
		if early_stopping.early_stop:
			print('Early stopping!!')
			break


	total_correct = 0
	model.load_state_dict(torch.load('../model/best_checkpoint.pt'))
	model.eval()
	with torch.no_grad():
		for i, data in enumerate(test_loader, 0):
			selected_data = (data[index] for index in data_index)
			pred_label = torch.squeeze(model(*selected_data).float())
			correct = ((pred_label > 0.5) == data[-1]).sum().item()
			total_correct += correct
					
		test_accuracy = float(total_correct) / len(test_loader.dataset)

		print('Accuracy of the network on the test dataset: %.2f%%' % ( 100.0 * float(total_correct) / len(test_loader.dataset)))
		compute_metrics(data[-1], pred_label.numpy(), verbose = True)

	return pred_label.unsqueeze(0)
		

######################################################################
		   		 #-- 10 times cross validation --#
######################################################################

def ten_times_cross_validation(train_pos_seqs, train_neg_seqs, 
							test_pos_seqs, test_neg_seqs,
							train_pos_shapes, train_neg_shapes, 
							test_pos_shapes, test_neg_shapes, layer_index):
	# read sequences to python list
	train_pos_seqs = read_fasta(train_pos_seqs)
	train_neg_seqs = read_fasta(train_neg_seqs)

	test_pos_seqs = read_fasta(test_pos_seqs)
	test_neg_seqs = read_fasta(test_neg_seqs)

	# One-hot
	train_pos_onehot = np.array(to_one_hot(train_pos_seqs)).astype(np.float32)
	train_neg_onehot = np.array(to_one_hot(train_neg_seqs)).astype(np.float32)

	test_pos_onehot = np.array(to_one_hot(test_pos_seqs)).astype(np.float32)
	test_neg_onehot = np.array(to_one_hot(test_neg_seqs)).astype(np.float32)

	train_onehot = np.concatenate((train_pos_onehot, train_neg_onehot), axis = 0)
	test_onehot = np.concatenate((test_pos_onehot, test_neg_onehot), axis = 0)

	# DNA shape
	train_shape = encode_shape(200, train_pos_shapes, train_neg_shapes)
	test_shape = encode_shape(200, test_pos_shapes, test_neg_shapes)

	# Kmer
	train_pos_kmer = np.array(seqs_to_kmers(train_pos_seqs)).astype(np.float32)
	train_neg_kmer = np.array(seqs_to_kmers(train_neg_seqs)).astype(np.float32)

	test_pos_kmer = np.array(seqs_to_kmers(test_pos_seqs)).astype(np.float32)
	test_neg_kmer = np.array(seqs_to_kmers(test_neg_seqs)).astype(np.float32)

	train_kmer = np.concatenate((train_pos_kmer, train_neg_kmer), axis = 0)
	test_kmer = np.concatenate((test_pos_kmer, test_neg_kmer), axis = 0)

	# generating labels
	train_label = [1] * int(len(train_kmer) / 2) + [0] * int(len(train_kmer) / 2)
	test_label = [1] * int(len(test_kmer) / 2) + [0] * int(len(test_kmer) / 2)
	train_label = np.array(train_label).astype(np.float32)
	test_label = np.array(test_label).astype(np.float32)

	random_index = random.sample(range(len(train_kmer)), len(train_kmer))
	train_onehot = train_onehot[random_index]
	train_kmer = train_kmer[random_index]
	train_shape = train_shape[random_index]
	train_label = train_label[random_index]

	train_onehot = torch.from_numpy(train_onehot).unsqueeze(1)
	test_onehot = torch.from_numpy(test_onehot).unsqueeze(1)

	train_shape = torch.from_numpy(train_shape)
	test_shape = torch.from_numpy(test_shape)

	train_kmer = torch.from_numpy(train_kmer)
	test_kmer = torch.from_numpy(test_kmer)

	train_label = torch.from_numpy(train_label)
	test_label = torch.from_numpy(test_label)


	for index in range(10):
		skf = StratifiedKFold(n_splits = 5, shuffle = True)

		fold_index = 1

		onehot_predicts = []
		shape_predicts = []
		kmer_predicts = []
		onehot_shape_predicts = []
		onehot_kmer_predicts = []
		shape_kmer_predicts = []
		onehot_shape_kmer_predicts = []

		all_predicts = []

		for train_index, val_index in skf.split(train_onehot, train_label):
			print('*' * 30 + ' fold ' + str(fold_index) + ' ' + '*' * 30)

			train_dataset = TensorDataset(train_onehot[train_index], train_shape[train_index], 
									train_kmer[train_index], train_label[train_index])
			train_loader = DataLoader(dataset = train_dataset, batch_size = 20, shuffle = True)

			val_dataset = TensorDataset(train_onehot[val_index], train_shape[val_index], 
									train_kmer[val_index], train_label[val_index])
			val_loader = DataLoader(dataset = val_dataset, batch_size = len(val_index), shuffle = True)

			test_dataset = TensorDataset(test_onehot, test_shape, test_kmer, test_label)
			test_loader = DataLoader(dataset = test_dataset, batch_size = layer_index * 200, shuffle = False)

			onehot_model = Onehot()
			shape_model = Shape()
			kmer_model = Kmer()
			onehot_shape_model = Onehot_Shape()
			onehot_kmer_model = Onehot_Kmer()
			shape_kmer_model = Shape_Kmer()
			onehot_shape_kmer_model = Onehot_Shape_Kmer()

			onehot_label = train_and_predict(train_loader, val_loader, test_loader, onehot_model, [0], 300, 1e-5, 30, 'mcc')
			shape_label = train_and_predict(train_loader, val_loader, test_loader, shape_model, [1], 300, 1e-5, 30, 'mcc')
			kmer_label = train_and_predict(train_loader, val_loader, test_loader, kmer_model, [2], 300, 1e-5, 30, 'mcc')
			onehot_shape_label = train_and_predict(train_loader, val_loader, test_loader, onehot_shape_model, [0, 1], 300, 1e-5, 30, 'mcc')
			onehot_kmer_label = train_and_predict(train_loader, val_loader, test_loader, onehot_kmer_model, [0, 2], 300, 1e-5, 30, 'mcc')
			shape_kmer_label = train_and_predict(train_loader, val_loader, test_loader, shape_kmer_model, [1, 2], 300, 1e-5, 30, 'mcc')
			onehot_shape_kmer_label = train_and_predict(train_loader, val_loader, test_loader, onehot_shape_kmer_model, [0, 1, 2], 300, 1e-5, 30, 'mcc')

			onehot_predicts.append(onehot_label)
			shape_predicts.append(shape_label)
			kmer_predicts.append(kmer_label)
			onehot_shape_predicts.append(onehot_shape_label)
			onehot_kmer_predicts.append(onehot_kmer_label)
			shape_kmer_predicts.append(shape_kmer_label)
			onehot_shape_kmer_predicts.append(onehot_shape_kmer_label)

			fold_index += 1

		all_predicts.append(onehot_predicts)
		all_predicts.append(shape_predicts)
		all_predicts.append(kmer_predicts)
		all_predicts.append(onehot_shape_predicts)
		all_predicts.append(onehot_kmer_predicts)
		all_predicts.append(shape_kmer_predicts)
		all_predicts.append(onehot_shape_kmer_predicts)

		feature_names = ['onehot', 'shape', 'kmer', 'onehot+shape', 'onehot+kmer', 'shape+kmer', 'all']

		print('*'*30 + 'FINAL RESULTS' + '*'*30)

		for i, predicts in enumerate(all_predicts):
			print('=' * 20 + feature_names[i]+ ' ' + '=' * 20)
			temp = torch.zeros(size = (1, 400))

			for labels in predicts:
				temp += labels

			temp = temp / 5

			compute_metrics(test_label, torch.squeeze(temp).numpy(), True)

			print('=' * 20 + feature_names[i]+ ' ' + '=' * 20 + '\n')

######################################################################
		   			  #-- main function --#
######################################################################

def main():
	usage = ''
	parser = OptionParser(usage)
	parser.add_option('-l', '--layer', default = False, type = 'int', help = 'layer index of enhancer prediction')
	(options, args) = parser.parse_args()
	layer_index = options.layer

	if layer_index == 1:
		# sequence files of first layer
		train_pos_seqs = '../data/layer1/train_enhancers.fa'
		train_neg_seqs = '../data/layer1/train_nonenhancers.fa'

		test_pos_seqs = '../data/layer1/test_enhancers.fa'
		test_neg_seqs = '../data/layer1/test_nonenhancers.fa'

		# DNA shape files of first layer
		train_pos_shapes = '../data/layer1/train_enhancers_13_shape.txt'
		train_neg_shapes = '../data/layer1/train_nonenhancers_13_shape.txt'

		test_pos_shapes = '../data/layer1/test_enhancers_13_shape.txt'
		test_neg_shapes = '../data/layer1/test_nonenhancers_13_shape.txt'

	elif layer_index == 2:
		# sequence files of second layer
		train_pos_seqs = '../data/layer2/train_strong_enhancers.fa'
		train_neg_seqs = '../data/layer2/train_weak_enhancers.fa'

		test_pos_seqs = '../data/layer2/test_strong_enhancers.fa'
		test_neg_seqs = '../data/layer2/test_weak_enhancers.fa'

		# DNA shape files of second layer
		train_pos_shapes = '../data/layer2/train_strong_enhancers_13_shape.txt'
		train_neg_shapes = '../data/layer2/train_weak_enhancers_13_shape.txt'

		test_pos_shapes = '../data/layer2/test_strong_enhancers_13_shape.txt'
		test_neg_shapes = '../data/layer2/test_weak_enhancers_13_shape.txt'

	ten_times_cross_validation(train_pos_seqs, train_neg_seqs, test_pos_seqs, test_neg_seqs,
						train_pos_shapes, train_neg_shapes, test_pos_shapes, test_neg_shapes, layer_index)


if __name__ == '__main__':
	main()
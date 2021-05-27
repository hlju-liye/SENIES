import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import auc

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from math import sqrt


# read DNA sequences to python list
def read_fasta(fasta_file_name):
	
	seqs = []
	seqs_num = 0
	file = open(fasta_file_name)

	for line in file.readlines():
		if line.strip() == '':
			continue

		if line.startswith('>'):
			seqs_num = seqs_num + 1
			continue
		else:
			seq = line.strip()
	
			result1 = 'N' in seq
			result2 = 'n' in seq
			if result1 == False and result2 == False:
				seqs.append(seq)
	return seqs

# formulate DNA sequences with one-hot encoding
def to_one_hot(seqs):
	base_dict = {
	'a' : 0, 'c' : 1, 'g' : 2, 't' : 3, 
	'A' : 0, 'C' : 1, 'G' : 2, 'T' : 3 
	}

	one_hot_4_seqs = []
	for seq in seqs:
		
		one_hot_matrix = np.zeros([4, len(seq)], dtype = float)
		index = 0
		for seq_base in seq:
			one_hot_matrix[base_dict[seq_base], index] = 1
			index = index + 1

		one_hot_4_seqs.append(one_hot_matrix)
	return one_hot_4_seqs

# formulate DNA sequences with kmer feature vectors
def seq_to_kmer(seq, k=5):
	encoding_matrix = {'a':0, 'A':0, 'c':1, 'C':1, 'g':2, 'G':2, 't':3, 'T':3, 'n':0, 'N':0}

	kmer_vec = np.zeros(4**k)

	for i in range(len(seq) - k + 1):
		sub_seq = seq[i:(i+k)]
		index = 0
		for j in range(k):
			index += encoding_matrix[sub_seq[j]]*(4**(k-j-1))
		kmer_vec[index] += 1
	return kmer_vec

def seqs_to_kmers(seqs):
	vecs = []
	for seq in seqs:
		temp = np.zeros(0)
		for i in range(5):
			temp = np.append(temp, seq_to_kmer(seq, i+1))
		vecs.append(temp)

	return vecs

# read DNA shape vectors from files output by DNAshapeR
def encode_shape(sample_len, pos_file_name, neg_file_name):

	shape = []
	for line in open(pos_file_name).readlines():
		content = line.strip().split()
		values = [float(value) for value in content]
		shape.append(values)

	for line in open(neg_file_name).readlines():
		content = line.strip().split()
		values = [float(value) for value in content]
		shape.append(values)
	
	return np.array(shape).astype(np.float32)

# compute five evaluation metrics from true labels and predicted labels
def compute_metrics(true_label, pre_label, verbose):
	fpr, tpr, thresholds = roc_curve(true_label, pre_label)
	AUC = auc(fpr, tpr)

	label_predict = (np.round(pre_label)).astype(np.int64)
	tn, fp, fn, tp = confusion_matrix(true_label, label_predict).ravel()

	tn = float(tn)
	fp = float(fp)
	fn = float(fn)
	tp = float(tp)
        
	SN = tp / (tp + fn)

	SP = tn / (tn + fp)

	ACC = (tp + tn) / (tp + tn + fn + fp)

	if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) == 0:
		MCC = -1
	else:
		MCC = (tp * tn - fp * fn) / sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

	if verbose:
		print('AUC = ' + str(AUC))

		print('ACC = ' + str(ACC))

		print('SN = ' + str(SN))

		print('SP = ' + str(SP))

		print('MCC = ' + str(MCC))

	return ACC, MCC

# class implementing the early stopping strategy
class EarlyStopping:
	def __init__(self, patience = 5, metrics = 'val_loss', verbose = False, delta = 0):

		self.patience = patience
		self.metrics = metrics
		self.verbose = verbose
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		self.val_metrics_best = np.Inf
		self.delta = delta

	def __call__(self, val_metrics, model):

		if self.metrics == 'val_loss':
			score = -val_metrics
		else:
			score = val_metrics

		if self.best_score is None:
			self.best_score = score
			self.save_checkpoint(val_metrics, model)
		elif score < self.best_score + self.delta:
			self.counter += 1
			if self.verbose:
				print('EarlyStopping counter: {%d} out of {%d}' % (self.counter, self.patience))

			if self.counter >= self.patience:
				self.early_stop = True
		else:
			self.best_score = score
			self.save_checkpoint(val_metrics, model)
			self.counter = 0

	def save_checkpoint(self, val_metrics, model):
		if self.verbose:
			#print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
			print('Saving model...')
		torch.save(model.state_dict(), '../model/best_checkpoint.pt')
		self.val_metrics_best = val_metrics


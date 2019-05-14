import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

class Visualizer:
	def __init__(self, histogram_intervals, top_wc_intervals):
		self.histogram_intervals = histogram_intervals
		self.top_wc_intervals = top_wc_intervals
		self.weak_classifier_accuracies = {}
		self.strong_classifier_scores = {}
		self.labels = None
	
	def draw_strong_errs(self):
		x = []
		y = []
		for t in self.strong_classifier_scores:
			x.append(t)
			cur_score = self.strong_classifier_scores[t]
			cur_predict = np.array([1 if p > 0 else -1 for p in cur_score])
			y.append(np.sum(1 if a != b else 0 for a, b in zip(cur_predict, self.labels)) / len(self.labels))
		plt.figure()
		plt.plot(x, y)
		plt.xlabel('Number of Weak Classifiers')
		plt.ylabel('Training Error')
		plt.title('Training Error Over Steps')
		plt.savefig('Training Error Over Steps')

	def draw_histograms(self):
		for t in self.strong_classifier_scores:
			if t not in [10, 50, 100]:
				continue
			scores = self.strong_classifier_scores[t]
			pos_scores = [scores[idx] for idx, label in enumerate(self.labels) if label == 1]
			neg_scores = [scores[idx] for idx, label in enumerate(self.labels) if label == -1]

			bins = np.linspace(np.min(scores), np.max(scores), 100)

			plt.figure()
			plt.hist(pos_scores, bins, alpha=0.5, label='Faces')
			plt.hist(neg_scores, bins, alpha=0.5, label='Non-Faces')
			plt.legend(loc='upper right')
			plt.title('Using %d Weak Classifiers' % t)
			plt.savefig('histogram_%d.png' % t)

	def draw_rocs(self):
		plt.figure()
		for t in self.strong_classifier_scores:
			if t not in [10, 50, 100]:
				continue
			scores = self.strong_classifier_scores[t]
			fpr, tpr, _ = roc_curve(self.labels, scores)
			plt.plot(fpr, tpr, label = 'No. %d Weak Classifiers' % t)
		plt.legend(loc = 'lower right')
		plt.title('ROC Curve')
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		plt.savefig('ROC Curve')

	def draw_wc_accuracies(self):
		plt.figure()
		for t in self.weak_classifier_accuracies:
			accuracies = self.weak_classifier_accuracies[t]
			plt.plot(accuracies, label = 'After %d Selection' % t)
		plt.ylabel('Accuracy')
		plt.xlabel('Weak Classifiers')
		plt.title('Top 1000 Weak Classifier Accuracies')
		plt.legend(loc = 'upper right')
		plt.savefig('Weak Classifier Accuracies')

if __name__ == '__main__':
	main()

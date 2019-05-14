import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle

import cv2
from weak_classifier import Ada_Weak_Classifier, Real_Weak_Classifier
from im_process import image2patches, nms, normalize


class Boosting_Classifier:
	def __init__(self, haar_filters, data, labels, num_chosen_wc, num_bins, visualizer, num_cores, style):
		self.filters = haar_filters
		self.data = data
		self.labels = labels
		self.num_chosen_wc = num_chosen_wc
		self.num_bins = num_bins
		self.visualizer = visualizer
		self.num_cores = num_cores
		self.style = style
		self.chosen_wcs = None
		if style == 'Ada':
			self.weak_classifiers = [Ada_Weak_Classifier(i, filt[0], filt[1], self.num_bins)\
									 for i, filt in enumerate(self.filters)]
		elif style == 'Real':
			self.weak_classifiers = [Real_Weak_Classifier(i, filt[0], filt[1], self.num_bins)\
									 for i, filt in enumerate(self.filters)]
	
	def calculate_training_activations(self, save_dir = None, load_dir = None):
		print('Calcuate activations for %d weak classifiers, using %d imags.' % (len(self.weak_classifiers), self.data.shape[0]))
		if load_dir is not None and os.path.exists(load_dir):
			print('[Find cached activations, %s loading...]' % load_dir)
			wc_activations = np.load(load_dir)
		else:
			if self.num_cores == 1:
				wc_activations = [wc.apply_filter(self.data) for wc in self.weak_classifiers]
			else:
				wc_activations = Parallel(n_jobs = self.num_cores)(delayed(wc.apply_filter)(self.data) for wc in self.weak_classifiers)
			wc_activations = np.array(wc_activations)
			if save_dir is not None:
				print('Writing results to disk...')
				np.save(save_dir, wc_activations)
				print('[Saved calculated activations to %s]' % save_dir)
		for wc in self.weak_classifiers:
			wc.activations = wc_activations[wc.id, :]
		return wc_activations
	
	#select weak classifiers to form a strong classifier
	#after training, by calling self.sc_function(), a prediction can be made
	#self.chosen_wcs should be assigned a value after self.train() finishes
	#call Weak_Classifier.calc_error() in this function
	#cache training results to self.visualizer for visualization
	#
	#
	#detailed implementation is up to you
	#consider caching partial results and using parallel computing
	def train(self, save_dir = None):
		######################
		######## TODO ########
		######################
		self.chosen_wcs = []
		weights = np.repeat(1.0 / len(self.data), len(self.data))
		strong_predict = np.zeros(len(self.data))
		for i in range(200):
			print(str(i))

			err_pol_thre = Parallel(n_jobs = self.num_cores)(delayed(wc.calc_error)(weights, self.labels) for wc in self.weak_classifiers)
			# err_pol_thre = Parallel(n_jobs = self.num_cores)(delayed(wc.calc_error)(weights, self.labels) for wc in self.weak_classifiers)
			# err_pol_thre = np.array([wc.calc_error(weights, self.labels) for wc in self.weak_classifiers])
			err_pol_thre = np.array(err_pol_thre)
			errs = err_pol_thre[:, 0]
			pols = err_pol_thre[:, 1]
			thres = err_pol_thre[:, 2]
			
			for idx in range(len(self.weak_classifiers)):
				self.weak_classifiers[idx].polarity = pols[idx]
				self.weak_classifiers[idx].threshold = thres[idx]
			
			# errs = np.array([wc.calc_error(weights, self.labels) for wc in self.weak_classifiers])
			
			if i in [0, 10, 50, 100]:
				acc = 1 - errs
				acc.sort()
				self.visualizer.weak_classifier_accuracies[i] = acc[::-1][:1000]

			if errs.min() <= 1e-7:
				break
			idx = np.argmin(errs)
			cur_err = errs[idx]
			cur_wc = self.weak_classifiers[idx]
			cur_alpha = 0.5 * np.log((1 - cur_err) / cur_err)
			self.chosen_wcs.append([cur_alpha, cur_wc])
			cur_predict = []
			for j in range(len(weights)):
				hj = cur_wc.polarity if cur_wc.activations[j] > cur_wc.threshold else -cur_wc.polarity
				weights[j] = weights[j] * np.exp(-1 * cur_alpha * self.labels[j] * hj)
				cur_predict.append(hj)
			weights = weights / np.sum(weights)
			strong_predict += cur_alpha * np.array(cur_predict)
			self.visualizer.strong_classifier_scores[i + 1] = strong_predict.copy()

		if save_dir is not None:
			pickle.dump(self.chosen_wcs, open(save_dir, 'wb'))


	def train_real_boost(self):
		self.load_trained_wcs('chosen_wcs.pkl')
		for t in [10, 50, 100]:
			# initial weak classifiers using step (c)
			self.weak_classifiers = [Real_Weak_Classifier(i, self.chosen_wcs[i][1].plus_rects, self.chosen_wcs[i][1].minus_rects, \
									self.num_bins) for i in range(t)]
			self.calculate_training_activations('wc_activations.npy', 'wc_activations.npy')
			# update t rounds to find best p and q arrays for each weak classifier
			weights = np.repeat(1.0 / len(self.data), len(self.data))
			for i in range(t):
				print(str(i))
				zs = np.array([wc.calc_error(weights, self.labels) for wc in self.weak_classifiers])
				if zs.min() <= 1e-7:
					break
				idx = np.argmin(zs)
				best_z = zs[idx]
				strong_predict = np.zeros(len(self.data))
				for j in range(len(weights)): # ht(xj)
					cur_predict = 0
					for cur_wc in self.weak_classifiers:
						bin_idx = np.sum(cur_wc.thresholds < cur_wc.activations[j])
						cur_predict += 0.5 * np.log(cur_wc.bin_pqs[0, bin_idx] / cur_wc.bin_pqs[1, bin_idx])
					weights[j] = (1 / best_z) * weights[j] * np.exp(-self.labels[j] * cur_predict)
					strong_predict[j] = cur_predict
				weights = weights / np.sum(weights)
			self.visualizer.strong_classifier_scores[t] = strong_predict.copy()


	def sc_function(self, image):
		return np.sum([np.array([alpha * wc.predict_image(image) for alpha, wc in self.chosen_wcs])])			

	def load_trained_wcs(self, save_dir):
		self.chosen_wcs = pickle.load(open(save_dir, 'rb'))	

	def face_detection(self, img, img_ori, scale_step = 10): ## 20
		
		# this training accuracy should be the same as your training process,
		##################################################################################
		train_predicts = []
		for idx in range(self.data.shape[0]):
			train_predicts.append(self.sc_function(self.data[idx, ...]))
		print('Check training accuracy is: ', np.mean(np.sign(train_predicts) == self.labels))
		##################################################################################

		# scales = 1 / np.linspace(1, 8, scale_step)
		scales = 1 / np.linspace(4, 8, scale_step)

		patches, patch_xyxy = image2patches(scales, img)
		print('Face Detection in Progress ..., total %d patches' % patches.shape[0])
		predicts = [self.sc_function(patch) for patch in tqdm(patches)]
		print(np.mean(np.array(predicts) > 0), np.sum(np.array(predicts) > 0))
		pos_predicts_xyxy = np.array([patch_xyxy[idx] + [score] for idx, score in enumerate(predicts) if score > 0])
		if pos_predicts_xyxy.shape[0] == 0:
			return
		xyxy_after_nms = nms(pos_predicts_xyxy, 0.01)
		
		print('after nms:', xyxy_after_nms.shape[0])
		for idx in range(xyxy_after_nms.shape[0]):
			pred = xyxy_after_nms[idx, :]
			cv2.rectangle(img_ori, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 255, 0), 10) #green rectangular with line width 3

		return img_ori

	def get_hard_negative_patches(self, img, scale_step = 10): ## 10
		# scales = 1 / np.linspace(1, 8, scale_step)
		scales = 1 / np.linspace(4, 8, scale_step)
		patches, patch_xyxy = image2patches(scales, img)
		print('Get Hard Negative in Progress ..., total %d patches' % patches.shape[0])
		predicts = np.array([self.sc_function(patch) for patch in tqdm(patches)])

		wrong_patches = patches[np.where(predicts > 0), ...]

		return wrong_patches

	def visualize(self):
		self.visualizer.labels = self.labels
		self.visualizer.draw_strong_errs()
		self.visualizer.draw_histograms()
		self.visualizer.draw_rocs()
		self.visualizer.draw_wc_accuracies()

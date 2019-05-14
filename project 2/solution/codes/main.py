import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt
import time
import cv2
from boosting_classifier import Boosting_Classifier
from visualizer import Visualizer
from im_process import normalize
from utils import *

def main():
	#flag for debugging
	flag_subset = False
	boosting_type = 'Ada' #'Real' or 'Ada'
	training_epochs = 100 if not flag_subset else 20
	act_cache_dir = 'wc_activations.npy' if not flag_subset else 'wc_activations_subset.npy'
	chosen_wc_cache_dir = 'chosen_wcs.pkl' if not flag_subset else 'chosen_wcs_subset.pkl'

	#data configurations
	pos_data_dir = '/Users/hannah_wang/Desktop/Release/newface16'
	neg_data_dir = '/Users/hannah_wang/Desktop/Release/nonface16'
	image_w = 16
	image_h = 16
	data, labels = load_data(pos_data_dir, neg_data_dir, image_w, image_h, flag_subset)
	data = integrate_images(normalize(data))

	#number of bins for boosting
	num_bins = 25

	#number of cpus for parallel computing
	num_cores = 8 if not flag_subset else 1 #always use 1 when debugging
	
	#create Haar filters
	filters = generate_Haar_filters(4, 4, 16, 16, image_w, image_h, flag_subset)

	#create visualizer to draw histograms, roc curves and best weak classifier accuracies
	drawer = Visualizer([10, 20, 50, 100], [1, 10, 20, 50, 100])
	
	#create boost classifier with a pool of weak classifier
	boost = Boosting_Classifier(filters, data, labels, training_epochs, num_bins, drawer, num_cores, boosting_type)

	#calculate filter values for all training images
	start = time.clock()
	boost.calculate_training_activations(act_cache_dir, act_cache_dir)
	end = time.clock()
	print('%f seconds for activation calculation' % (end - start))

	# (a) display top 20 haar filters after boosting
	boost.train(chosen_wc_cache_dir)

	sorted_haar_filters = sorted(boost.chosen_wcs, key=itemgetter(0))[::-1]
	plt.figure(figsize=(16, 14))
	plt.suptitle('Top 20 Haar Filters', fontsize=24,x=0.5,y=0.93)
	for i in range(20):
		print(sorted_haar_filters[i][0])
		ax = plt.subplot(4, 5, 1+i)
		plt.axis([0, 16, 0, 16])
		ax.get_xaxis().set_visible(False) 
		ax.get_yaxis().set_visible(False)
		plus_rects = sorted_haar_filters[i][1].plus_rects
		minus_rects = sorted_haar_filters[i][1].minus_rects
		current_axis=plt.gca()
		for plus_rect in plus_rects:
			rect = plt.Rectangle((plus_rect[0], plus_rect[1]),plus_rect[2]-plus_rect[0]+1,plus_rect[3]-plus_rect[1]+1,\
									linewidth=1,edgecolor='k',facecolor='none')
			current_axis.add_patch(rect)
		for minus_rect in minus_rects:
			rect = plt.Rectangle((minus_rect[0], minus_rect[1]),minus_rect[2]-minus_rect[0]+1,minus_rect[3]-minus_rect[1]+1,\
									linewidth=1,edgecolor='k',facecolor='k')
			current_axis.add_patch(rect)
		
	plt.show()
    
	# (b ~ e)
	boost.visualize()

	# (f) 
	for i in range(3):
		original_img = cv2.imread('/Users/hannah_wang/Desktop/Release/Test Images 2018/Test ' + str(i+1) + '.jpeg')
		original_img_gray = cv2.imread('/Users/hannah_wang/Desktop/Release/Test Images 2018/Test ' + str(i+1) + '.jpeg', cv2.IMREAD_GRAYSCALE)
		result_img = boost.face_detection(original_img_gray, original_img)
		plt.imshow(result_img)
		cv2.imwrite('Result_img_' + str(i+1), result_img)

	# (g)
	# add hard negatives to training set
	for i in range(2):
		neg_img = cv2.imread('/Users/hannah_wang/Desktop/Release/Test Images 2018/Hard Negative ' + str(i+1) + '.jpeg', cv2.IMREAD_GRAYSCALE)
		hard_negative_patches = boost.get_hard_negative_patches(neg_img)

		data = np.concatenate((data, hard_negative_patches[0]))
		labels = np.concatenate((labels, -1 * np.ones(len(hard_negative_patches[0]))))

	# re-train model and detect faces again

	# (h ~ i)
	boost.train_real_boost()
	boost.visualize()

if __name__ == '__main__':
	main()


# coding: utf-8

# In[177]:


import matplotlib.pyplot as plt 
import matplotlib.image as img
import numpy as np
from skimage import color
from sklearn.decomposition import PCA
import os
import scipy
import math
import mywarper


# ============ preprocessing: index -> img_name, index -> landmarks ============ #
def read_files(path):
    file_list = os.listdir(path)
    file_list = [path + file for file in file_list]
    file_list.sort()
    return file_list


# # Question 1

# In[178]:


num_train = 800
num_test = 200
num_eigenfaces = 10
img_list = read_files("./data/images/")
train_face_matrix = np.zeros((128*128, 800))
train_face_matrix_central = np.zeros((128*128, 800))
test_face_matrix_origin = []
test_face_matrix_central = np.zeros((128*128, 200))

# ============ 2.1 Q1.1 compute mean face ============= #
mean_face = np.zeros((128, 128))
for i in range(num_train):
    image = color.rgb2hsv(img.imread(img_list[i]))[:, :, 2]
    mean_face += image
    train_face_matrix[:, i] = np.reshape(image, (128*128))

mean_face /= num_train
mean_face_vector = np.reshape(mean_face, (128*128))
plt.imshow(mean_face, cmap='Greys_r') # show grey image (only channel v)
plt.axis('off') 
plt.title('Mean Face (Channel V)', fontsize=18)
plt.show()

# centralize train_face_matrix
for i in range(num_train):
    train_face_matrix_central[:, i] = train_face_matrix[:, i] - mean_face_vector


# In[179]:


# ============ 2.1 Q1.2 compute eigen-faces ============= #
pca = PCA(n_components=50)
pca.fit(train_face_matrix_central.transpose())
eigen_vectors = pca.components_.transpose()

# display first 10 eigen-faces
plt.figure(figsize=(16, 7))
plt.suptitle('First 10 Eigen-faces', fontsize=24,x=0.5,y=0.97)
for i in range(num_eigenfaces):
    eigen_face_img = np.reshape(eigen_vectors[:, i], (128, 128))
    plt.subplot(2, 5, 1+i)
    plt.axis('off')
    plt.title('Eigen Face ' + str(i+1), fontsize=16)
    plt.imshow(eigen_face_img, cmap='Greys_r')
plt.subplots_adjust(wspace=0.2, hspace=0)
plt.show()


# In[180]:


# ============ 2.1 Q1.3 reconstruct 10 faces ============= #
# read and centralize test images
mean_face_matrix = np.zeros((128*128, 200))
for i in range(num_test):
    mean_face_matrix[:, i] += mean_face_vector
    image = img.imread(img_list[800+i])
    test_face_matrix_origin.append(image)
    image_hsv = color.rgb2hsv(image)[:, :, 2]
    test_face_matrix_central[:, i] = np.reshape(image_hsv, (128*128)) - mean_face_vector

# project test faces to eigen-vectors
beta = np.dot(test_face_matrix_central.transpose(), eigen_vectors)
reconstruct_faces = np.dot(eigen_vectors, beta.transpose()) + mean_face_matrix

# plot first 10 reconstructed faces and their original faces
plt.figure(figsize=(16, 14))
plt.suptitle('First 10 Reconstructed Faces', fontsize=24, x=0.5, y=0.92)
for i in range(num_eigenfaces):
    # plot reconstructed image
    hsv_img = color.rgb2hsv(test_face_matrix_origin[i])
    hsv_img[:, :, 2] = np.reshape(reconstruct_faces[:, i], (128, 128))
    recons_image = color.hsv2rgb(hsv_img)
    pos = 1 + i
    if i >= 5:
        pos = 6 + i
    plt.subplot(4, 5, pos)
    plt.axis('off')
    plt.title('Reconstructed Face ' + str(i+1), fontsize=12)
    plt.imshow(color.hsv2rgb(hsv_img))
    # plot original image
    plt.subplot(4, 5, pos+5)
    plt.axis('off')
    plt.title('Original Face ' + str(i+1), fontsize=12)
    plt.imshow(test_face_matrix_origin[i])
plt.subplots_adjust(wspace=0.2, hspace=0)
plt.show()


# In[181]:


# ============ 2.1 Q1.4 plot reconstruction error ============= #
# extract original channel v
test_face_matrix_v = np.zeros((128*128, 200))
for i in range(num_test):
    hsv_img = color.rgb2hsv(test_face_matrix_origin[i])
    test_face_matrix_v[:, i] += np.reshape(hsv_img[:, :, 2], (128*128))

# calculate reconstruction error over k
k_values = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
errors = []
for k in k_values:
    # compute k eigen-faces
    pca = PCA(n_components=k)
    pca.fit(train_face_matrix_central.transpose())
    eigen_vectors = pca.components_.transpose()
    
    # reconstruct 200 test faces
    beta = np.dot(test_face_matrix_central.transpose(), eigen_vectors)
    reconstruct_faces = np.dot(eigen_vectors, beta.transpose()) + mean_face_matrix
    
    # normalize reconstructed faces to [0, 1]
    for i in range(num_test):
        cur_img = reconstruct_faces[:, i]
        min_v = cur_img.min()
        max_v = cur_img.max()
        cur_img -= min_v * np.ones((128*128))
        cur_img /= (max_v - min_v)
        reconstruct_faces[:, i] = cur_img
    
    # calculate total error
    error = np.power(reconstruct_faces - test_face_matrix_v, 2).sum() / (128*128*200)
    errors.append(error)

# plot reconstruction error
plt.figure(figsize=(16,9))
plt.xlabel('Number of Eigen-faces K', fontsize = 18)
plt.ylabel('Total Reconstruction Error', fontsize = 18)
plt.title('Reconstruction Error vs K', fontsize = 23)
plt.plot(k_values, errors, 'o-', color='deeppink', lw=2)
plt.show()


# # Question 2

# In[182]:


landmark_list = read_files("./data/landmarks/")
train_landmark_matrix = np.zeros((68*2, 800))
train_landmark_matrix_central = np.zeros((68*2, 800))
test_landmark_matrix_origin = []
test_landmark_matrix_central = np.zeros((68*2, 200))

# ============ 2.1 Q2.1 compute mean landmark ============= #
mean_landmark = np.zeros((68, 2))
for i in range(num_train):
    landmark = scipy.io.loadmat(landmark_list[i])['lms']
    mean_landmark += landmark
    train_landmark_matrix[:, i] = np.reshape(landmark, (68*2))

mean_landmark /= num_train
mean_landmark_vector = np.reshape(mean_landmark, (68*2))
plt.imshow(mean_face, cmap='Greys_r') # show grey image (only channel v)
plt.scatter(mean_landmark[:, 0], mean_landmark[:, 1], color='deeppink', lw=0.5)
plt.axis('off') 
plt.title('Mean Landmark on Mean Face', fontsize=18)
plt.show()

# centralize train_landmark_matrix
for i in range(num_train):
    train_landmark_matrix_central[:, i] = train_landmark_matrix[:, i] - mean_landmark_vector


# In[183]:


# ============ 2.1 Q2.2 compute eigen-warpings ============= #
pca = PCA(n_components=10)
pca.fit(train_landmark_matrix_central.transpose())
eigen_warpings = pca.components_.transpose()

# display first 10 eigen-warpings
line_color=['deepskyblue', 'coral', 'limegreen', 'gold', 'c', 'deeppink', 'mediumorchid', 'lightpink', 'olive', 'red']
plt.figure(1, figsize=(18, 8))
plt.suptitle('First 10 Eigen-warpings', fontsize=24, x=0.5, y=0.97)
plt.figure(2, figsize=(8, 8))
for i in range(num_eigenfaces):
    eigen_warping_img = np.reshape(eigen_warpings[:, i]+mean_landmark_vector, (68, 2))
    plt.figure(1)
    plt.subplot(2, 5, 1+i)
    plt.axis('off')
    plt.title('Eigen Warping ' + str(i+1), fontsize=16)
    plt.imshow(mean_face, cmap='Greys_r')
    plt.scatter(eigen_warping_img[:, 0], eigen_warping_img[:, 1], color=line_color[i], lw=0.5)
    
    plt.figure(2)
    plt.axis('off')
    plt.title('10 Eigen Warpings on One Mean Face', fontsize=22)
    plt.scatter(eigen_warping_img[:, 0], eigen_warping_img[:, 1], color=line_color[i], lw=0.5)
    plt.imshow(mean_face, cmap='Greys_r')
plt.subplots_adjust(wspace=0.2, hspace=0)
plt.show()


# In[184]:


# ============ 2.1 Q2.3 reconstruct 10 landmarks ============= #
# read and centralize test landmarks
mean_landmark_matrix = np.zeros((68*2, 200))
for i in range(num_test):
    mean_landmark_matrix[:, i] += mean_landmark_vector
    landmark = scipy.io.loadmat(landmark_list[800+i])['lms']
    test_landmark_matrix_origin.append(landmark)
    test_landmark_matrix_central[:, i] = np.reshape(landmark, (68*2)) - mean_landmark_vector

# project test landmarks to eigen-warpings
beta = np.dot(test_landmark_matrix_central.transpose(), eigen_warpings)
reconstruct_landmarks = np.dot(eigen_warpings, beta.transpose()) + mean_landmark_matrix

# plot first 10 reconstructed landmarks and their original landmarks
plt.figure(figsize=(18, 8))
plt.suptitle('First 10 Reconstructed Landmarks vs Original Landmarks', fontsize=24, x=0.5, y=0.95)
for i in range(num_eigenfaces):
    # plot original landmark
    plt.subplot(2, 5, 1+i)
    plt.axis('off')
    plt.title('Reconstructed Landmark ' + str(i+1), fontsize=12)
    plt.imshow(test_face_matrix_origin[i])
    orig_landmark = test_landmark_matrix_origin[i]
    l1 = plt.scatter(orig_landmark[:, 0], orig_landmark[:, 1], color='deepskyblue', lw=0.1)
    # plot reconstructed landmark
    recons_landmark = reconstruct_landmarks[:, i].reshape((68, 2))
    l2 = plt.scatter(recons_landmark[:, 0], recons_landmark[:, 1], color='gold', lw=0.1)
plt.subplots_adjust(wspace=0.2, hspace=0)
plt.legend(handles=[l1, l2], labels=['Original Landmarks', 'Reconstructed Landmarks'],  bbox_to_anchor=(-4.85,-0.2), loc='center left')
plt.show()


# In[185]:


# ============ 2.1 Q2.4 plot reconstruction error ============= #
# calculate reconstruction error over k
errors = []
for k in range(10):
    # compute k eigen-warpings
    pca = PCA(n_components=k+1)
    pca.fit(train_landmark_matrix_central.transpose())
    eigen_warpings = pca.components_.transpose()
    
    # reconstruct 200 test landmarks
    beta = np.dot(test_landmark_matrix_central.transpose(), eigen_warpings)
    reconstruct_landmarks = np.dot(eigen_warpings, beta.transpose()) + mean_landmark_matrix
    
    # calculate total error
    error = 0
    for i in range(200):
        orig_landmark = test_landmark_matrix_origin[i]
        recons_landmark = reconstruct_landmarks[:, i].reshape((68, 2))
        for j in range(68):
            orig_point = orig_landmark[j]
            recons_point = recons_landmark[j]
            error += math.sqrt(np.power(orig_point[0] - recons_point[0], 2) + np.power(orig_point[1] - recons_point[1], 2))
    error /= 200 * 68
    errors.append(error)

# plot reconstruction error
plt.figure(figsize=(16,9))
plt.xlabel('Number of Eigen-warpings K', fontsize = 18)
plt.ylabel('Total Reconstruction Error', fontsize = 18)
plt.title('Reconstruction Error vs K', fontsize = 23)
plt.plot([i+1 for i in range(10)], errors, 'o-', color='deeppink', lw=2)
plt.show()


# # Question 3

# In[187]:


train_aligned_face_matrix = np.zeros((128*128, 800))
train_aligned_face_matrix_central = np.zeros((128*128, 800))
mean_aligned_face = np.zeros((128, 128))

# ============ 2.1 Q3.1 align training image to mean position ============= #
for i in range(num_train):
    image = img.imread(img_list[i])
    ori_landmark = scipy.io.loadmat(landmark_list[i])['lms']
    warped_face = mywarper.warp(image, ori_landmark, mean_landmark)
    warped_face_v = color.rgb2hsv(warped_face)[:, :, 2]
    mean_aligned_face += warped_face_v
    train_aligned_face_matrix[:, i] = warped_face_v.reshape((128*128))

mean_aligned_face /= num_train
mean_aligned_face_vector = mean_aligned_face.reshape((128*128))

# centralize train_aligned_face_matrix
for i in range(num_train):
    train_aligned_face_matrix_central[:, i] = train_aligned_face_matrix[:, i] - mean_aligned_face_vector    

plt.imshow(mean_aligned_face, cmap='Greys_r') # show grey image (only channel v)
plt.axis('off') 
plt.title('Mean Aligned Face (Channel V)', fontsize=18)
plt.show()
plt.imshow(warped_face)


# In[188]:


# ============ 2.1 Q3.2 compute eigen-faces of aligned faces ============= #
pca = PCA(n_components=50)
pca.fit(train_aligned_face_matrix_central.transpose())
aligned_eigen_vectors = pca.components_.transpose()

# display first 10 aligned eigen-faces
plt.figure(figsize=(16, 7))
plt.suptitle('First 10 Aligned Eigen-faces', fontsize=24,x=0.5,y=0.97)
for i in range(num_eigenfaces):
    eigen_face_img = np.reshape(aligned_eigen_vectors[:, i], (128, 128))
    plt.subplot(2, 5, 1+i)
    plt.axis('off')
    plt.title('Aligned Eigen Face ' + str(i+1), fontsize=14)
    plt.imshow(eigen_face_img, cmap='Greys_r')
plt.subplots_adjust(wspace=0.2, hspace=0)
plt.show()


# In[189]:


# ============ 2.1 Q3.3 reconstruct landmarks of test data ============= #
# use results from Q2 reconstruct_landmarks (136x200)

test_aligned_face_hsv = []
test_aligned_face_matrix_central = np.zeros((128*128, 200))
mean_aligned_face_matrix = np.zeros((128*128, 200))

# ============ 2.1 Q3.4 warp test image to mean position ============= #
for i in range(num_test):
    mean_aligned_face_matrix[:, i] += mean_aligned_face_vector
    image = img.imread(img_list[800+i])
    ori_landmark = scipy.io.loadmat(landmark_list[800+i])['lms']
    warped_face = mywarper.warp(image, ori_landmark, mean_landmark)
    warped_face_hsv = color.rgb2hsv(warped_face)
    test_aligned_face_hsv.append(warped_face_hsv)
    cur_aligned_face = warped_face_hsv[:, :, 2].reshape((128*128))
    test_aligned_face_matrix_central[:, i] = cur_aligned_face - mean_aligned_face_vector    


# In[213]:


# ============ 2.1 Q3.5 reconstruct test faces at mean position ============= #
# project aligned test faces to aligned eigen-vectors
aligned_beta = np.dot(test_aligned_face_matrix_central.transpose(), aligned_eigen_vectors)
reconstruct_aligned_faces = np.dot(aligned_eigen_vectors, aligned_beta.transpose()) + mean_aligned_face_matrix

# ============ 2.1 Q3.6 warp reconstructed test faces to reconstructed landmarks ============= #
# plot first 20 reconstructed faces and their original faces
plt.figure(figsize=(16, 27))
plt.suptitle('First 20 Reconstructed Faces (with Warping)', fontsize=24, x=0.5, y=0.9)
for i in range(20):
    # plot reconstructed image
    hsv_img = test_aligned_face_hsv[i]
    hsv_img[:, :, 2] = np.reshape(reconstruct_aligned_faces[:, i], (128, 128))
    recons_rgb_image = color.hsv2rgb(hsv_img)
    recons_image = mywarper.warp(recons_rgb_image, mean_landmark, reconstruct_landmarks[:, i].reshape((68, 2)))
    pos = 1 + i
    if i >= 15:
        pos = 16 + i
    elif i >= 10:
        pos = 11 + i
    elif i >= 5:
        pos = 6 + i
    plt.subplot(8, 5, pos)
    plt.axis('off')
    plt.title('Reconstructed Face ' + str(i+1), fontsize=12)
    plt.imshow(recons_image)
    # plot original image
    plt.subplot(8, 5, pos+5)
    plt.axis('off')
    plt.title('Original Face ' + str(i+1), fontsize=12)
    plt.imshow(test_face_matrix_origin[i])
plt.subplots_adjust(wspace=0.2, hspace=0)
plt.show()


# In[214]:


# ============ 2.1 Q3.7 plot reconstruction error ============= #
# calculate reconstruction error over k
k_values = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
errors_with_warping = []
for k in k_values:
    # compute k eigen-faces
    pca = PCA(n_components=k)
    pca.fit(train_aligned_face_matrix_central.transpose())
    aligned_eigen_vectors = pca.components_.transpose()
    
    # reconstruct 200 test faces
    aligned_beta = np.dot(test_aligned_face_matrix_central.transpose(), aligned_eigen_vectors)
    reconstruct_aligned_faces = np.dot(aligned_eigen_vectors, aligned_beta.transpose()) + mean_aligned_face_matrix

    # warp back
    final_recons_faces_v = np.zeros((128*128, 200))
    for i in range(num_test):
        hsv_img = test_aligned_face_hsv[i]
        hsv_img[:, :, 2] = np.reshape(reconstruct_aligned_faces[:, i], (128, 128))
        recons_rgb_image = color.hsv2rgb(hsv_img)
        recons_image = mywarper.warp(recons_rgb_image, mean_landmark, reconstruct_landmarks[:, i].reshape((68, 2)))
        recons_image_v = color.rgb2hsv(recons_image)[:, :, 2].reshape((128*128))
        # normalize reconstructed faces to [0, 1]
        min_v = recons_image_v.min()
        max_v = recons_image_v.max()
        recons_image_v -= min_v * np.ones((128*128))
        recons_image_v /= (max_v - min_v)
        final_recons_faces_v[:, i] = recons_image_v
    
    # calculate total error
    error = np.power(final_recons_faces_v - test_face_matrix_v, 2).sum() / (128*128*200)
    errors_with_warping.append(error)

# plot reconstruction error
plt.figure(figsize=(16,9))
plt.xlabel('Number of Eigen-faces K', fontsize = 18)
plt.ylabel('Total Reconstruction Error', fontsize = 18)
plt.title('Reconstruction Error vs K (with Warping)', fontsize = 23)
plt.plot(k_values, errors_with_warping, 'o-', color='deeppink', lw=2)
plt.show()


# # Question 4

# In[220]:


# ============ 2.1 Q4.1 get two bases ============= #
pca_face = PCA(n_components=50)
pca_face.fit(train_aligned_face_matrix_central.transpose())
aligned_eigen_faces = pca_face.components_.transpose()

pca_landmark = PCA(n_components=10)
pca_landmark.fit(train_landmark_matrix_central.transpose())
eigen_warpings = pca_landmark.components_.transpose()

# ============ 2.1 Q4.2 generate coefficient by normal distribution and plot 50 random faces ============= #
sqrt_eigen_faces = np.sqrt(pca_face.explained_variance_) # sqrt(eigen values)
sqrt_eigen_warpings = np.sqrt(pca_landmark.explained_variance_)

plt.figure(figsize=(16,30))
plt.suptitle('50 Random Synthesized Faces (with Warping)', fontsize=24, x=0.5, y=0.9)
for i in range(50):
    random_coef_face = np.random.randn(50) * sqrt_eigen_faces
    random_coef_landmark = np.random.randn(10) * sqrt_eigen_warpings

    random_recons_aligned_faces = np.dot(aligned_eigen_faces, random_coef_face) + mean_aligned_face_vector
    random_recons_aligned_faces = random_recons_aligned_faces.reshape((128, 128))

    random_recons_landmarks = np.dot(eigen_warpings, random_coef_landmark) + mean_landmark_vector
    random_recons_face = mywarper.warp(random_recons_aligned_faces.reshape(128, 128, 1), mean_landmark, random_recons_landmarks.reshape((68, 2)))

    plt.subplot(10, 5, 1+i)
    plt.imshow(random_recons_face.reshape(128, 128), cmap='Greys_r')
    plt.title('Random Face ' + str(i+1), fontsize=12)
    plt.axis('off')
plt.show()


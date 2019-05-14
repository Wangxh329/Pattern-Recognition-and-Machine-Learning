% prep
clear all;
close all;

% flags
flag_compile_libsvm_c = 0;
flag_compile_libsvm_mex = 0;

% compile libsvm
if flag_compile_libsvm_c
    parent = cd('libsvm-3.21');
    [status,cmdout] = system('make');
    cd(parent);
    disp(status);
    disp(cmdout);
end

if flag_compile_libsvm_mex
    parent = cd('libsvm-3.21/matlab');
    make;
    cd(parent);
end

% setup
diary('P1_2.out');
rng(123);
addpath('libsvm-3.21/matlab');

% data
disp('loading data ...');
load('hog/hog_sbin_32.mat', 'hog_img');
load('train-anno.mat', 'face_landmark', 'trait_annotation');

features = face_landmark;
labels = trait_annotation;
num_samples = size(face_landmark, 1);
mean_trait = mean(trait_annotation, 1);

% Hogfeat = [];
% for i = 1:491
%     impath = ['./img/M', sprintf('%04d.jpg', i)];
%     im = double(imread(impath));
%     hogfeat = HoGfeatures(im);
%     Hogfeat = [Hogfeat; hogfeat(:)'];
% end

del = find((max(hog_img, [], 1) - min(hog_img, [], 1)) == 0);
hog_img(:, del) = [];
hog_img = double(hog_img);
HogfeatScale = bsxfun(@minus, hog_img', min(hog_img, [], 1)');
HogfeatScale = bsxfun(@times, HogfeatScale', 1./max(hog_img, [], 2));
hog_img = HogfeatScale;

face_landmark = [face_landmark(:, 1:76) face_landmark(:, 81:end-4)];
face_landmark = normc(face_landmark);

% 10-fold cross-validation
flag_cv = 1;

if flag_cv
    learningTime = tic; 
    all_features = [face_landmark, hog_img];
    [C,Gamma,Epsilon] = meshgrid(-3:2:7, -19:2:-5, -9:2:1);
    error = zeros(numel(C), 1);
    gamma = zeros(14, 1);
    cost = zeros(14, 1);
    epsilon = zeros(14, 1);
    minError = zeros(14, 1);
    
    for t = 1:14
        fprintf('%d  iteration: ', t);
        label = labels(:,t);
        for i = 1:numel(C)
            display([t Gamma(i) C(i) Epsilon(i)]);
            cmd = sprintf('-s 3 -t 2 -v 5 -q -g %0.5f -c %0.5f -p %0.5f', 2.^Gamma(i), 2.^C(i), 2.^Epsilon(i));
            error(i) = libsvmtrain(label, all_features, cmd);
        end
        fprintf('cross validation took %0.2f\n\n\n\n');
        [minError(t), I] = min(error);
        gamma(t) = Gamma(I);
        cost(t) = C(I);
        epsilon(t) = Epsilon(I);
    end
    learningTime = toc(learningTime);
    save('p1_2.mat', 'epsilon', 'gamma', 'cost', 'minError', '-v7.3');
else
    load('p1_2.mat', 'epsilon', 'gamma', 'cost', 'minError');
end

% predict
disp('cross validation ...');
folds = 10;
acc_rec_test = zeros(folds, 14);
acc_rec_train = zeros(folds, 14);
prec_rec_test = zeros(folds, 14);
prec_rec_train = zeros(folds, 14);

for i = 1:folds
    div = floor(num_samples/folds);
    start = (i-1) * div + 1;
    endd = i * div;
    
    train_lm = [face_landmark(1:start-1, :); face_landmark(endd+1:end, :)];
    train_label = [labels(1:start-1, :); labels(endd+1:end, :)];
    train_hog = [hog_img(1:start-1, :); hog_img(endd+1:end, :)];
    train_features = [train_lm, train_hog];
    
    test_lm = face_landmark(start:endd, :);
    test_label = labels(start:endd, :);
    test_hog = hog_img(start:endd, :);
    test_features = [test_lm, test_hog];
    
    fprintf('%d cross-validation: \n\n', i);
    for j = 1:14
        fprintf('%d  iteration \n', j);
        g = gamma(j);
        c = cost(j);
        p = epsilon(j);
        
        cmd = sprintf('-s 3 -t 2 -b 1 -q -g %0.5f -c %0.5f -p %0.5f', 2.^g, 2.^c, 2.^p);
        models{i, j} = libsvmtrain(train_label(:, j), train_features, cmd);
        
        [predict_label, ~, ~] = libsvmpredict(test_label(:, j), test_features, models{i, j}, '-q '); 
        true_pos = sum((predict_label>=0) .* (test_label(:, j)>=0));
        false_pos = sum((predict_label>=0) .* (test_label(:, j)<0));
        prec_rec_test(i, j) = true_pos / (false_pos + true_pos);
        if (false_pos + true_pos) == 0
            prec_rec_test(i, j) = 0;
        end
        acc_rec_test(i, j) = sum((predict_label >= 0) == (test_label(:, j) >= 0)) / size(test_label, 1);
        
        [predict_label, ~, ~] = libsvmpredict(train_label(:, j), train_lm, models{i, j}, '-q'); 
        true_pos = sum((predict_label>=0) .* (train_label(:,j)>=0));
        false_pos = sum((predict_label>=0) .* (train_label(:,j)<0));
        prec_rec_train(i,j) = true_pos / (false_pos + true_pos);
        if (false_pos + true_pos) == 0
            prec_rec_train(i, j) = 0;
        end
        acc_rec_train(i,j) = sum((predict_label >= 0) == (train_label(:, j)>=0)) / size(train_label, 1);
    end
end

avg_acc_test = mean(acc_rec_test, 1);
avg_acc_train = mean(acc_rec_train, 1);
avg_prec_test = mean(prec_rec_test, 1);
avg_prec_train = mean(prec_rec_train, 1);

save('P1_2_result.mat', 'models', 'acc_rec_test', 'acc_rec_train', 'prec_rec_test', 'prec_rec_train', 'avg_prec_train', 'avg_prec_test', 'avg_acc_train', 'avg_acc_test');

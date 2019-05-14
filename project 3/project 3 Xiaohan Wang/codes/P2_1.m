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
diary('P2_1.out');
rng(123);
addpath('libsvm-3.21/matlab');

% data
disp('loading data ...');
load('hog/hog_sbin_32.mat');

load('stat-gov.mat');
gov_lm = face_landmark;
vote_diff_gov = vote_diff;

load('stat-sen.mat');
sen_lm = face_landmark;
vote_diff_sen = vote_diff;

% classification
for class = 1:2
    if class == 1
        face_lm = gov_lm;
        vote_diff = vote_diff_gov;
        hog_sbin32 = hog_elec_gov;
    end

    if class == 2
        face_lm = sen_lm;
        vote_diff = vote_diff_sen;
        hog_sbin32 = hog_elec_sen;
    end
    
    num_samples = size(face_lm, 1);

    labels = vote_diff;
    features = [face_lm, hog_sbin32];
    features = normc(features);

    % predict
    disp('cross validation ...');
    folds = 10;
    acc_rec_test = zeros(folds, 1);
    acc_rec_train = zeros(folds, 1);

    models = cell(folds, 1);

    Cost = -15:1:15;
    avg_acc_test = 0;

    for j = 1:numel(Cost)
        for i = 1:folds
            num_test = floor(num_samples / folds);
            start = (i-1) * num_test + 1;
            endd = i * num_test;

            train_features = [features(1:start-1, :); features(endd+1:end, :)];
            train_labels = [labels(1:start-1); labels(endd+1:end)];

            test_features = features(start:endd, :);
            test_labels = labels(start:endd);

            cmd = sprintf('-s 3 -c %0.5f', 2.^Cost(j));

            models{i} = libsvmtrain(train_labels, train_features, cmd);
            [predict_label, ~, ~] = libsvmpredict(test_labels, test_features, models{i}, '-q'); 
            acc_rec_test(i) = sum((predict_label >= 0) == (test_labels>=0)) / size(test_labels, 1);

            [predict_label, ~, ~] = libsvmpredict(train_labels, train_features, models{i}, '-q'); 
            acc_rec_train(i) = sum((predict_label >= 0) == (train_labels>=0)) / size(train_labels, 1);    
        end

        if mean(acc_rec_test,1) > avg_acc_test || avg_acc_test == 0        
            avg_acc_test = mean(acc_rec_test, 1);
            avg_acc_train = mean(acc_rec_train, 1);

            j_max_acc = j;
        end
    end

    Cost_chosen = Cost(j_max_acc);
    if class == 1
        save('P2_1_gov.mat', 'models', 'acc_rec_test', 'acc_rec_train', 'avg_acc_train', 'avg_acc_test', 'Cost_chosen');
    else
        save('P2_1_sen.mat', 'models', 'acc_rec_test', 'acc_rec_train', 'avg_acc_train', 'avg_acc_test', 'Cost_chosen');
    end
end

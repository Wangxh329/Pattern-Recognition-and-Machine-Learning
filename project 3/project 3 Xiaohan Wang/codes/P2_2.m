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
diary('P2_2.out');
rng(123);
addpath('libsvm-3.21/matlab');

% load svm
disp('loading svm models ...');
load('P1_2_result.mat');

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
        face_landmark = gov_lm;
        vote_diff = vote_diff_gov;
        hog_sbin32 = hog_elec_gov;
    end

    if class == 2
        face_landmark = sen_lm;
        vote_diff = vote_diff_gov;
        hog_sbin32 = hog_elec_sen;
    end

    num_samples = size(face_landmark, 1);

    labels = vote_diff;
    features = [face_landmark, hog_sbin32];
    features = normc(features);

    % predict
    disp('cross validation ...');
    folds = 5;
    acc_rec_test = zeros(folds, 1);
    acc_rec_train = zeros(folds, 1);

    new_model = cell(folds, 1);
    avg_acc_test = 0;
    Cost = -15:2:15;

    for c = 1:numel(Cost)
        fprintf('Cost %d   \n', c);

        for i = 1:folds
            num_test = floor(num_samples / folds);
            start = (i-1) * num_test + 1;
            endd = i * num_test;

            outcome = -ones(num_samples, 1);
            outcome(1:2:end) = 1;
            
            trait = zeros(num_samples, 14);
            for j = 1:14
                [trait(:,j), ~, ~] = libsvmpredict(zeros(size(features, 1), 1), features, models{i, j}, '-q'); 
            end

            trait_diff = zeros(num_samples, 14);  % unbiased
            for j = 1:(num_samples / 2)
                trait_diff(j*2-1, :) = trait(j*2, :) - trait(j*2-1, :);
                trait_diff(j*2, :) = trait(j*2-1, :) - trait(j*2, :);
            end

            train_trait = vertcat(trait_diff(1:start-1, :), trait_diff(endd+1:end, :));
            train_labels = vertcat(outcome(1:start-1), outcome(endd+1:end));   
            test_trait = trait_diff(start:endd, :);
            test_labels = outcome(start:endd);

            cmd = sprintf('-c %0.5f', 2.^Cost(c));
            new_model{i} = libsvmtrain(train_labels, train_trait, cmd);

            [predict_label, ~, ~] = libsvmpredict(test_labels, test_trait, new_model{i}, '-q'); 
            accuracy(i) = sum((predict_label >= 0) == (test_labels>=0)) / size(test_labels, 1);

            [predict_label, ~, ~] = libsvmpredict(train_labels, train_trait, new_model{i}, '-q');
            acc_rec_train(i) = sum((predict_label >= 0) == (train_labels>=0)) / size(train_labels,1);
        end

        if mean(accuracy) > avg_acc_test || avg_acc_test == 0    
            cor = corr(trait_diff, outcome);
            avg_acc_test = mean(accuracy);
            avg_acc_train = mean(acc_rec_train);

            j_max_acc = c;
        end
    end

    Cost_chosen = Cost(j_max_acc);
    if class == 1
        save('P2_2_gov.mat', 'new_model', 'acc_rec_test', 'acc_rec_train', 'avg_acc_train', 'avg_acc_test','cor', 'Cost_chosen');
    else
        save('P2_2_sen.mat', 'new_model', 'acc_rec_test', 'acc_rec_train', 'avg_acc_train', 'avg_acc_test','cor', 'Cost_chosen');
    end
end

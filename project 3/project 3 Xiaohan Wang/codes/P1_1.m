% prep
clear all;
close all;

% flags
flag_compile_libsvm_c = 1;
flag_compile_libsvm_mex = 1;

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
end

% setup
diary('P1_1.out');
rng(123);
addpath('libsvm-3.21/matlab');

% data
disp('loading data ...');
load('train-anno.mat', 'face_landmark', 'trait_annotation');
features = [face_landmark(:,1:76) face_landmark(:,81:end-4)];
labels = trait_annotation;

% predict
disp('cross validation ...');
%% todo: perform k-fold cross-validation

% grid search for model parameters
flag_grid = 1;

if flag_grid
    learningTime = tic; 
    [C, Gamma, Epsilon] = meshgrid(-7:2:7, -19:2:-5, -11:2:1);
    model = zeros(numel(C), 1);
    gamma = zeros(14, 1);
    cost = zeros(14, 1);
    epsilon = zeros(14, 1);
    minError = zeros(14, 1);
    
    for i = 1:14
        fprintf('%d  iteration: ', i);
        for i = 1:numel(C)
            display([i Gamma(i) C(i) Epsilon(i)]);
            cmd = sprintf('-s 3 -t 2 -v 5 -q -g %0.5f -c %0.5f -p %0.5f',...
                2.^Gamma(i), 2.^C(i), 2.^Epsilon(i));
            model(i) = libsvmtrain(labels(:, i), features, cmd);
        end
        [minError(i), idx] = min(model);
        gamma(i) = Gamma(idx);
        cost(i) = C(idx);
        epsilon(i) = Epsilon(idx);
    end
    learningTime = toc(learningTime);
    save('p1_1.mat', 'epsilon', 'gamma', 'cost', 'minError', '-v7.3');
else
    load('p1_1.mat', 'epsilon', 'gamma', 'cost', 'minError');
end

% 10-fold cross-validation for training and testing 
folds = 10;
acc_train = zeros(folds, 14);
acc_test = zeros(folds, 14);
precision_train = zeros(folds, 14);
precision_test = zeros(folds, 14);

for fold = 1:folds
    [trainId, ~, testIdx] = dividerand(491, 1-1/folds, 0, 1/folds);
    fea_train = features(trainIndex, :);
    fea_test = features(testIdx, :);
    label_train = labels(trainIndex, :);
    label_test = labels(testIdx, :);
    
    for i = 1:14
        g = gamma(i);
        c = cost(i);
        p = epsilon(i);

        cmd = sprintf('-s 3 -t 2 -b 1 -q -g %0.5f -c %0.5f -p %0.5f', 2.^g, 2.^c, 2.^p);
        model = libsvmtrain(label_train(:,i), fea_train, cmd);

        [PredictTrain,TrainAcc,~] = libsvmpredict(label_train(:,i), fea_train, model);
        true_pos = sum((PredictTrain >= 0).*(label_train(:,i) >= 0));
        false_pos = sum((PredictTrain >= 0) .*(label_train(:,i) < 0));
        precision_train(fold, i) = true_pos / (true_pos + false_pos);
        if (true_pos + false_pos) == 0
            precision_train(fold, i) = 0;
        end 
        acc_train(fold, i) = sum((PredictTrain >= 0) == (label_train(:, i) >= 0)) / size(label_train(:, i), 1);
        
        [PredictTest, TestAcc, ~] = libsvmpredict(label_test(:, i), fea_test, model);
        true_pos = sum((PredictTest>=0).*(label_test(:,i) >=0));
        false_pos = sum((PredictTest>=0).*(label_test(:,i) <0));
        precision_test(fold, i) = true_pos / (true_pos + false_pos);
        if (true_pos + false_pos) == 0
            precision_test(fold, i) = 0;
        end 
        acc_test(fold, i) = sum((PredictTest >= 0) == (label_test(:, i) >= 0)) / size(label_test(:, i), 1);
        
    end
end

avg_acc_train = mean(acc_train, 1);
avg_acc_test = mean(acc_test, 1);

avg_prec_train = mean(precision_train, 1);
avg_prec_test = mean(precision_test, 1);

save('p1_1_acc_pre.mat', 'acc_train', 'acc_test', 'precision_train', ...
    'precision_test', 'avg_acc_train', 'avg_acc_test', 'avg_prec_train', 'avg_prec_test', '-v7.3');

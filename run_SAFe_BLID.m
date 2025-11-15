% run_SAFe_BLID.m
% Main runner for SAFe-BLID pipeline (MATLAB R2015a compatible)
clear; close all; clc;
rng(42); % reproducible seed

% ========== CONFIG ==========
cfg.dataset = 'UNSW'; % 'UNSW' | 'CICIDS2017' | 'KDD'
cfg.data_folder = './data/';
cfg.results_folder = './results/';
if ~exist(cfg.results_folder,'dir'), mkdir(cfg.results_folder); end

cfg.numIoT = 800;        % simulated IoT devices (10..1000)
cfg.ga_pop = 40;         % GA population
cfg.ga_gen = 25;         % GA generations (raise if you have time)
cfg.alpha = 0.8; cfg.beta = 0.2; % fitness weights
cfg.mu_max = 0.2;        % max mutation
cfg.elitism = 2;
cfg.crossover_rate = 0.8;

cfg.classifier_for_ga = 'mlp'; % 'mlp' or 'svm' (fast approx in GA)
cfg.final_train_epochs = 30;   % final HSA-BiLSTM epochs (if Deep Learning Toolbox available)
cfg.batch_size = 128;
cfg.lr = 0.001;
cfg.sequence_length = 10; % T

cfg.use_oversample = true; % simple random oversampling

% ========== LOAD DATASET ==========
fprintf('Loading dataset %s...\n', cfg.dataset);
[X, Y, feature_names] = load_dataset(cfg.dataset, cfg.data_folder);
[n, m] = size(X);
fprintf('Loaded %d samples with %d features\n', n, m);

% ========== SIMULATE IOT METADATA (OPTIONAL) ==========
iot_meta = simulate_iot_devices(cfg.numIoT);

% ========== PREPROCESS ==========
fprintf('Preprocessing data...\n');
[Xp, Yp, preproc_info] = preprocess_data(X, Y, cfg);
save(fullfile(cfg.results_folder,'preproc_info.mat'),'preproc_info');

% ========== FEATURE SELECTION (EMOGFS) ==========
fprintf('Running EMOGFS (GA feature selection)...\n');
[best_mask, ga_log] = EMOGFS(Xp, Yp, cfg, feature_names);
sel_features = find(best_mask==1);
fprintf('Selected %d features out of %d\n', numel(sel_features), m);
save(fullfile(cfg.results_folder,'ga_log.mat'),'ga_log','best_mask','sel_features');

% ========== CREATE SEQUENCES ==========
fprintf('Creating sequences (T=%d)...\n', cfg.sequence_length);
[Xseq, Yseq] = create_sequences(Xp(:,sel_features), Yp, cfg.sequence_length);

% ========== TRAIN FINAL HSA-BiLSTM (or fallback MLP) ==========
fprintf('Training classifier (HSA-BiLSTM if available)...\n');
train_opts.epochs = cfg.final_train_epochs;
train_opts.batch = cfg.batch_size;
train_opts.lr = cfg.lr;
train_opts.hidden = [128, 64];
train_opts.attention = true;
train_opts.useDLToolbox = check_DL_toolbox();

[model, history] = train_HSA_BiLSTM(Xseq, Yseq, train_opts);

% ========== EVALUATE ==========
[metrics, confmat, Ypred] = evaluate_model(model, Xseq, Yseq, train_opts);
fprintf('--- Final Evaluation ---\n');
fprintf('Accuracy: %.3f %%\n', metrics.Accuracy*100);
fprintf('Precision: %.3f %%\n', metrics.Precision*100);
fprintf('Recall: %.3f %%\n', metrics.Recall*100);
fprintf('F1-score: %.3f %%\n', metrics.F1*100);
fprintf('FPR: %.3f %%\n', metrics.FPR*100);

% Save results
save(fullfile(cfg.results_folder,'final_results.mat'),'metrics','confmat','history','sel_features','model','Ypred');

fprintf('All done. Results saved to %s\n', cfg.results_folder);

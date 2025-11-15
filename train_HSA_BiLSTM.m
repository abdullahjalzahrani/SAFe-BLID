function [model, history] = train_HSA_BiLSTM(Xseq, Yseq, opts)
% Xseq: [N x T x D], Yseq: [N x 1]
useDL = opts.useDLToolbox;
[N, T, D] = size(Xseq);
classes = unique(Yseq);
K = numel(classes);

if useDL
    % Prepare cell array of sequences (D x T) each
    Xcell = cell(N,1);
    for i=1:N, Xcell{i} = squeeze(Xseq(i,:,:))'; end % D x T
    Ycat = categorical(Yseq);
    layers = [ ...
        sequenceInputLayer(D,'Name','input')
        bilstmLayer(opts.hidden(1),'OutputMode','sequence','Name','bilstm1')
        bilstmLayer(opts.hidden(2),'OutputMode','last','Name','bilstm2')
        fullyConnectedLayer(64,'Name','fc1')
        reluLayer('Name','relu1')
        dropoutLayer(0.5,'Name','drop1')
        fullyConnectedLayer(K,'Name','fc_out')
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    options = trainingOptions('adam', ...
        'InitialLearnRate',opts.lr, ...
        'MaxEpochs',opts.epochs, ...
        'MiniBatchSize',opts.batch, ...
        'Shuffle','every-epoch', ...
        'Verbose',true, ...
        'Plots','training-progress');
    tbl = array2table(1:N','VariableNames',{'idx'});
    % trainNetwork accepts cell arrays and categorical labels directly
    try
        net = trainNetwork(Xcell, Ycat, layers, options);
        model.type = 'dltoolbox';
        model.net = net;
        history = []; % training plot visible during run
    catch ME
        warning('trainNetwork failed: %s\nFalling back to MLP.', ME.message);
        useDL = false;
    end
end

if ~useDL
    % Flatten sequences: each sample -> vector of length T*D
    Xflat = reshape(Xseq, N, T*D);
    % simple MLP using patternnet if available
    if exist('patternnet','file')
        net = patternnet(100);
        net.trainParam.epochs = opts.epochs;
        net.trainParam.showWindow = false;
        net = train(net, Xflat', ind2vec(Yseq') );
        model.type = 'mlp';
        model.net = net;
        history = [];
    else
        % fallback: fitcecoc (SVM) or simple linear model
        if exist('fitcecoc','file')
            Mdl = fitcecoc(Xflat, Yseq);
            model.type = 'svm';
            model.net = Mdl;
            history = [];
        else
            error('No suitable classifier available on this MATLAB installation.');
        end
    end
end
end

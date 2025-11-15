function [metrics, confmat, Ypred] = evaluate_model(model, Xseq, Ytrue, opts)
% Evaluate the trained model; returns metrics struct and confusion matrix
[N, T, D] = size(Xseq);
% Predict depending on model.type
switch model.type
    case 'dltoolbox'
        Xcell = cell(N,1);
        for i=1:N, Xcell{i} = squeeze(Xseq(i,:,:))'; end
        YpredCat = classify(model.net, Xcell, 'MiniBatchSize', opts.batch);
        Ypred = double(YpredCat);
    case 'mlp'
        Xflat = reshape(Xseq, N, T*D);
        out = model.net(Xflat');
        [~, idx] = max(out, [], 1);
        Ypred = idx';
    case 'svm'
        Xflat = reshape(Xseq, N, T*D);
        Ypred = predict(model.net, Xflat);
    otherwise
        error('Unknown model.type %s', model.type);
end

[acc, prec, rec, f1, fpr, CM] = compute_metrics(Ytrue, Ypred);
metrics.Accuracy = acc;
metrics.Precision = prec;
metrics.Recall = rec;
metrics.F1 = f1;
metrics.FPR = fpr;
confmat = CM;
end

% compute_metrics: returns acc, precision (macro), recall (macro), f1 (macro), fpr (macro), confusion matrix
function [acc, prec, rec, f1, fpr, CM] = compute_metrics(Ytrue, Ypred)
classes = unique(Ytrue);
K = numel(classes);
CM = zeros(K,K);
for i=1:K
    for j=1:K
        CM(i,j) = sum((Ytrue==classes(i)) & (Ypred==classes(j)));
    end
end
TPs = diag(CM);
acc = sum(TPs)/sum(CM(:));
prec_per = zeros(K,1); rec_per = zeros(K,1); f1_per = zeros(K,1); fpr_per = zeros(K,1);
for i=1:K
    TP = CM(i,i);
    FP = sum(CM(:,i)) - TP;
    FN = sum(CM(i,:)) - TP;
    TN = sum(CM(:)) - TP - FP - FN;
    if TP+FP==0, prec_per(i)=0; else prec_per(i)=TP/(TP+FP); end
    if TP+FN==0, rec_per(i)=0; else rec_per(i)=TP/(TP+FN); end
    if prec_per(i)+rec_per(i)==0, f1_per(i)=0; else f1_per(i)=2*prec_per(i)*rec_per(i)/(prec_per(i)+rec_per(i)); end
    if FP+TN==0, fpr_per(i)=0; else fpr_per(i)=FP/(FP+TN); end
end
prec = mean(prec_per); rec = mean(rec_per); f1 = mean(f1_per); fpr = mean(fpr_per);
end

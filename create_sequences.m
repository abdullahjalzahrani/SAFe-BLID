function [Xseq, Yseq] = create_sequences(X, Y, T)
% Convert sample-by-feature into sequences using non-overlapping windows of length T
[n, d] = size(X);
numSeq = floor(n / T);
Xseq = zeros(numSeq, T, d);
Yseq = zeros(numSeq,1);
for i=1:numSeq
    idx = (i-1)*T + (1:T);
    seg = X(idx,:);
    Xseq(i,:,:) = seg;
    Yseq(i) = mode(Y(idx)); % label by majority in window
end
end

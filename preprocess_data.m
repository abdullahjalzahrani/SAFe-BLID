function [Xn, Yn, info] = preprocess_data(X, Y, cfg)
% Missing imputation (mean), min-max normalization, optional oversampling.
X = double(X);
[n,m] = size(X);

% impute NaN with column mean
for j=1:m
    col = X(:,j);
    if any(isnan(col))
        col(isnan(col)) = nanmean(col);
        X(:,j) = col;
    end
end

% min-max normalize
minv = min(X,[],1);
maxv = max(X,[],1);
den = maxv - minv; den(den==0)=1;
Xn = bsxfun(@rdivide, bsxfun(@minus, X, minv), den);

Yn = Y;

% optional oversample (simple random replicate)
if cfg.use_oversample
    cls = unique(Yn);
    counts = arrayfun(@(c)sum(Yn==c), cls);
    maxc = max(counts);
    Xout = []; Yout = [];
    for i=1:numel(cls)
        idx = find(Yn==cls(i));
        Xout = [Xout; Xn(idx,:)];
        Yout = [Yout; Yn(idx)];
        if numel(idx) < maxc
            r = randi(numel(idx), maxc - numel(idx), 1);
            Xout = [Xout; Xn(idx(r),:)];
            Yout = [Yout; Yn(idx(r))];
        end
    end
    Xn = Xout; Yn = Yout;
end

info.minv = minv; info.maxv = maxv; info.n = size(Xn,1);
end

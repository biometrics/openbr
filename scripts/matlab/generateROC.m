function [ta fa] = genROC_ME(S,rMat)
% [ta fa] = genROC_ME(S,rMat)
% S is score similarity matrix 
%
% rMat is a matrix contain the relationship of each image pair
%   - 0 for different subject
%   - 1 for same subject
%   - 2 to skip the pair

    if nargin < 2,
        rMat = uint8(eye(size(S)));
    end
    S = S(rMat ~=2);
    rMat = rMat(rMat ~=2);

    nTrue = sum(rMat(:) == 1);
    nFalse = numel(rMat) - nTrue;

    [S1 order] = sort(S(:),'descend');

    n = numel(S1);

    rank = zeros(nTrue,1);
    t_vals = zeros(nTrue,1);

    cnt = 0;
    for i = 1:n,
        if rMat(order(i)) == 1
            cnt = cnt + 1;

            t_vals(cnt) = S1(order(i));
            rank(cnt) = i - cnt;
        end
    end

    ta = zeros(nTrue,1);
    fa = zeros(nTrue,1);
    for i = 1:length(t_vals),
        ta(i) = i/nTrue;
        fa(i) = rank(i)/nFalse;
    end
    aaa=1;
end

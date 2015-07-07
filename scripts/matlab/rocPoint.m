function [ta score] = getROC(t,f,th)
% [ta] = getROC(t,f,th)
    [a1 a2] = min(abs(f - th));
    ta = t(a2);
    score = a2;
    if max(f) == 0
        ta = 1.0;
    end

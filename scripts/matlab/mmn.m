function [S1] = mmn(S)
S1 = (S - min(S(:))) ./ (max(S(:)) - min(S(:)));
end

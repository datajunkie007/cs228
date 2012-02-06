%COMPUTEEXACTMARGINALSBP Runs exact inference and returns the marginals
%over all the variables (if isMax == 0) or the max-marginals (if isMax == 1). 
%
%   M = COMPUTEEXACTMARGINALSBP(F, E, isMax) takes a list of factors F,
%   evidence E, and a flag isMax, runs exact inference and returns the
%   final marginals for the variables in the network. If isMax is 1, then
%   it runs exact MAP inference, otherwise exact inference (sum-prod).
%   It returns an array of size equal to the number of variables in the 
%   network where M(i) represents the ith variable and M(i).val represents 
%   the marginals of the ith variable. 


% CS228 Probabilistic Models in AI (Winter 2012)
% Copyright (C) 2012, Stanford University

function M = ComputeExactMarginalsBP(F, E, isMax)

% Since we only need marginals at the end, you should M as:
%
% M = repmat(struct('var', 0, 'card', 0, 'val', []), length(N), 1);
%
% where N is the number of variables in the network, which can be determined
% from the factors F.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% Implement Exact and MAP Inference.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

P = CreateCliqueTree(F, E);

P = CliqueTreeCalibrate(P, isMax);

N = length(F);
vars = [];

for i = 1:N
  vars = union(vars, F(i).var);
end

M = repmat(struct('var', 0, 'card', 0, 'val', []), length(vars), 1);

for i = 1:length(vars)
  v = vars(i);
  for j = 1:length(P.cliqueList)
    if ismember(v, P.cliqueList(j).var)
        if isMax
          M(v) = FactorMaxMarginalization(P.cliqueList(j), setdiff(P.cliqueList(j).var, [v]));
        else
          M(v) = FactorMarginalization(P.cliqueList(j), setdiff(P.cliqueList(j).var, [v]));
          M(v).val = M(v).val ./ sum(M(v).val); % normalization
        end
      break;
    end
  end
end

end

%CLIQUETREECALIBRATE Performs sum-product or max-product algorithm for 
%clique tree calibration.

%   P = CLIQUETREECALIBRATE(P, isMax) calibrates a given clique tree, P 
%   according to the value of isMax flag. If isMax is 1, it uses max-sum
%   message passing, otherwise uses sum-product. This function 
%   returns the clique tree where the .val for each clique in .cliqueList
%   is set to the final calibrated potentials.

% CS228 Probabilistic Models in AI (Winter 2012)
% Copyright (C) 2012, Stanford University

function P = CliqueTreeCalibrate(P, isMax)


% Number of cliques in the tree.
N = length(P.cliqueList);

% Setting up the messages that will be passed.
% MESSAGES(i,j) represents the message going from clique i to clique j. 
MESSAGES = repmat(struct('var', [], 'card', [], 'val', []), N, N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% While there are ready cliques to pass messages between, keep passing
% messages. Use GetNextCliques to find cliques to pass messages between.
% Once you have clique i that is ready to send message to clique
% j, compute the message and put it in MESSAGES(i,j).
% Remember that you only need an upward pass and a downward pass.
%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[nexti, nextj] = GetNextCliques(P, MESSAGES);

while nexti && nextj
  sepset = intersect(P.cliqueList(nexti).var, P.cliqueList(nextj).var);
  tempf = P.cliqueList(nexti);
  for k = 1:N
    if k ~= nextj
      tempf = FactorProduct(tempf, MESSAGES(k, nexti));
    end
  end
  MESSAGES(nexti, nextj) = FactorMarginalization(tempf, setdiff(tempf.var, sepset));
  MESSAGES(nexti, nextj).val = MESSAGES(nexti, nextj).val ./ sum(MESSAGES(nexti, nextj).val);
  [nexti, nextj] = GetNextCliques(P, MESSAGES);  

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% Now the clique tree has been calibrated. 
% Compute the final potentials for the cliques and place them in P.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:N
  for k = 1:N
    if P.edges(i,k)
      P.cliqueList(i) = FactorProduct(P.cliqueList(i),MESSAGES(k,i));
    end
  end
end

return

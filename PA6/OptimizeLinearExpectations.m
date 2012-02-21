function [MEU OptimalDecisionRule] = OptimizeLinearExpectations( I )
% Inputs: An influence diagram I with a single decision node and one or more utility nodes.
%         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
%              the child variable = D.var(1)
%         I.DecisionFactors = factor for the decision node.
%         I.UtilityFactors = list of factors representing conditional utilities.
% Return value: the maximum expected utility of I and an optimal decision rule 
% (represented again as a factor) that yields that expected utility.
% You may assume that there is a unique optimal decision.
%
% This is similar to OptimizeMEU except that we will have to account for
% multiple utility factors.  We will do this by calculating the expected
% utility factors and combining them, then optimizing with respect to that
% combined expected utility factor.  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% YOUR CODE HERE
%
% A decision rule for D assigns, for each joint assignment to D's parents, 
% probability 1 to the best option from the EUF for that joint assignment 
% to D's parents, and 0 otherwise.  Note that when D has no parents, it is
% a degenerate case we can handle separately for convenience.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

EUF = struct('var', [], 'card', [], 'val', []);

for i = 1:length(I.UtilityFactors)
  tmpI = I;
  tmpI.UtilityFactors = I.UtilityFactors(i);
  tmpEUF = CalculateExpectedUtilityFactor(tmpI);
  EUF = FactorSum(EUF, tmpEUF);
end

OptimalDecisionRule = struct('var', [], 'card', [], 'val', []);
OptimalDecisionRule.var = EUF.var;
OptimalDecisionRule.card = EUF.card;
OptimalDecisionRule.val = zeros(prod(OptimalDecisionRule.card), 1);
if length(EUF.var) < 2,
  [MEU myIndex] = max(EUF.val);
  OptimalDecisionRule.val(myIndex) = 1;
else
  MEU = 0.0;
  fullAssignment = IndexToAssignment([1:prod(OptimalDecisionRule.card)],OptimalDecisionRule.card);
  for i = 1:prod(OptimalDecisionRule.card(2:end))
      subAssignment = IndexToAssignment(i,OptimalDecisionRule.card(2:end));
      myIndex = [];
      for j = 1:size(fullAssignment,1)
          if all(fullAssignment(j,2:size(fullAssignment,2))==subAssignment)
              myIndex = [myIndex;j];
          end
      end
      Assignment = IndexToAssignment(myIndex,OptimalDecisionRule.card);
      myValue = GetValueOfAssignment(EUF,Assignment);
      [myMax mySubIndex] = max(myValue);
      OptimalDecisionRule.val(myIndex) = 0;
      OptimalDecisionRule.val(myIndex(mySubIndex)) = 1;
      MEU = MEU + EUF.val(myIndex(mySubIndex));
  end
end

return;

end

function C = FactorSum(A, B)

% Check for empty factors
if (isempty(A.var)), C = B; return; end;
if (isempty(B.var)), C = A; return; end;

% Check that variables in both A and B have the same cardinality
[dummy iA iB] = intersect(A.var, B.var);
if ~isempty(dummy)
  % A and B have at least 1 variable in common
  assert(all(A.card(iA) == B.card(iB)), 'Dimensionality mismatch in factors');
end

% Set the variables of C
C.var = union(A.var, B.var);

% Construct the mapping between variables in A and B and variables in C.
% In the code below, we have that
%
%   mapA(i) = j, if and only if, A.var(i) == C.var(j)
% 
% and similarly 
%
%   mapB(i) = j, if and only if, B.var(i) == C.var(j)
%
% For example, if A.var = [3 1 4], B.var = [4 5], and C.var = [1 3 4 5],
% then, mapA = [2 1 3] and mapB = [3 4]; mapA(1) = 2 because A.var(1) = 3
% and C.var(2) = 3, so A.var(1) == C.var(2).

[dummy, mapA] = ismember(A.var, C.var);
[dummy, mapB] = ismember(B.var, C.var);

% Set the cardinality of variables in C
C.card = zeros(1, length(C.var));
C.card(mapA) = A.card;
C.card(mapB) = B.card;

% Initialize the factor values of C:
%   prod(C.card) is the number of entries in C
C.val = zeros(1,prod(C.card));

% Compute some helper indices
% These will be very useful for calculating C.val
% so make sure you understand what these lines are doing.
assignments = IndexToAssignment(1:prod(C.card), C.card);
indxA = AssignmentToIndex(assignments(:, mapA), A.card);
indxB = AssignmentToIndex(assignments(:, mapB), B.card);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE:
% Correctly populate the factor values of C
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C.val = A.val(indxA) + B.val(indxB);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

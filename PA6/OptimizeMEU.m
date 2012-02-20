function [MEU OptimalDecisionRule] = OptimizeMEU( I )

  % Inputs: An influence diagram I with a single decision node and a single utility node.
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return value: the maximum expected utility of I and an optimal decision rule 
  % (represented again as a factor) that yields that expected utility.
  
  % We assume I has a single decision node.
  % You may assume that there is a unique optimal decision.
  D = I.DecisionFactors(1);

  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % YOUR CODE HERE...
  % 
  % Some other information that might be useful for some implementations
  % (note that there are multiple ways to implement this):
  % 1.  It is probably easiest to think of two cases - D has parents and D 
  %     has no parents.
  % 2.  You may find the Matlab/Octave function setdiff useful.
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    EUF = CalculateExpectedUtilityFactor(I);
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
%           myIndex = find(all(fullAssignment(:,2:size(fullAssignment,2))==subAssignment));
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
        

end

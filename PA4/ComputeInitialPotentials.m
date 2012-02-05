%COMPUTEINITIALPOTENTIALS Sets up the cliques in the clique tree that is
%passed in as a parameter.
%
%   P = COMPUTEINITIALPOTENTIALS(C) Takes the clique tree skeleton C which is a
%   struct with three fields:
%   - nodes: cell array representing the cliques in the tree.
%   - edges: represents the adjacency matrix of the tree.
%   - factorList: represents the list of factors that were used to build
%   the tree. 
%   
%   It returns the standard form of a clique tree P that we will use through 
%   the rest of the assigment. P is struct with two fields:
%   - cliqueList: represents an array of cliques with appropriate factors 
%   from factorList assigned to each clique. Where the .val of each clique
%   is initialized to the initial potential of that clique.
%   - edges: represents the adjacency matrix of the tree. 


% CS228 Probabilistic Models in AI (Winter 2012)
% Copyright (C) 2012, Stanford University

function P = ComputeInitialPotentials(C)

% number of cliques
N = length(C.nodes);

% initialize cluster potentials 
P.cliqueList = repmat(struct('var', [], 'card', [], 'val', []), N, 1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% First, compute an assignment of factors from factorList to cliques. 
% Then use that assignment to initialize the cliques in cliqueList to 
% their initial potentials. 

% Hint: C.nodes is a list of cliques.
% P.cliqueList(i).var = C.nodes{i};
% Print out C to get a better understanding of its structure.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M = length(C.factorList);
assigned = zeros(M, 1);

for i = 1:N

  P.cliqueList(i).var = C.nodes{i};

  for j = 1:M
    if (~assigned(j))
      [tf, index] = ismember (C.factorList(j).var, P.cliqueList(i).var);
      if (all(tf))
        P.cliqueList(i).card(index) = C.factorList(j).card;
        assigned(j) = i;
      end
    end
  end

  P.cliqueList(i).val = ones(1, prod(P.cliqueList(i).card));

end

for j = 1:M
  P.cliqueList(assigned(j)).val = FactorProduct(P.cliqueList(assigned(j)),C.factorList(j)).val;
end

P.edges = C.edges;

end


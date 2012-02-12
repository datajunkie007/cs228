% CLUSTERGRAPHCALIBRATE Loopy belief propagation for cluster graph calibration.
%   P = CLUSTERGRAPHCALIBRATE(P, useSmart) calibrates a given cluster graph, G,
%   and set of of factors, F. The function returns the final potentials for
%   each cluster. 
%   The cluster graph data structure has the following fields:
%   - .clusterList: a list of the cluster beliefs in this graph. These entries
%                   have the following subfields:
%     - .var:  indices of variables in the specified cluster
%     - .card: cardinality of variables in the specified cluster
%     - .val:  the cluster's beliefs about these variables
%   - .edges: A cluster adjacency matrix where edges(i,j)=1 implies clusters i
%             and j share an edge.
%  
%   UseSmart is an indicator variable that tells us whether to use the Naive or Smart
%   implementation of GetNextClusters for our message ordering
%
%   See also FACTORPRODUCT, FACTORMARGINALIZATION

% CS228 Probabilistic Models in AI (Winter 2012)
% Copyright (C) 2012, Stanford University

function [P MESSAGES] = ClusterGraphCalibrate(P,useSmartMP)

if(~exist('useSmartMP','var'))
  useSmartMP = 0;
end

N = length(P.clusterList);

MESSAGES = repmat(struct('var', [], 'card', [], 'val', []), N, N);
[edgeFromIndx, edgeToIndx] = find(P.edges ~= 0);

for m = 1:length(edgeFromIndx),
    i = edgeFromIndx(m);
    j = edgeToIndx(m);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    %
    %
    %
    % Set the initial message values
    % MESSAGES(i,j) should be set to the initial value for the
    % message from cluster i to cluster j
    %
    % The matlab/octave functions 'intersect' and 'find' may
    % be useful here (for making your code faster)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [ MESSAGES(i,j).var, ia, ib ] = intersect(P.clusterList(i).var, P.clusterList(j).var);
    MESSAGES(i,j).card = P.clusterList(i).card(ia);
    MESSAGES(i,j).val = ones(1, prod(MESSAGES(i,j).card));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end;



% perform loopy belief propagation
tic;
iteration = 0;

lastMESSAGES = MESSAGES;

% for quiz question
thresh = 1.0e-6;
ijij = [ 19 3; 15 40; 17 2 ];
X = []; Y = []; Z = [];

while (1),
    iteration = iteration + 1;
    [i, j] = GetNextClusters(P, MESSAGES, lastMESSAGES, iteration, useSmartMP); 
    prevMessage = MESSAGES(i,j);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    % We have already selected a message to pass, \delta_ij.
    % Compute the message from clique i to clique j and put it
    % in MESSAGES(i,j)
    % Finally, normalize the message to prevent overflow
    %
    % The function 'setdiff' may be useful to help you
    % obtain some speedup in this function
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    sepset = intersect(P.clusterList(i).var, P.clusterList(j).var);
    tempf = P.clusterList(i);
    for k = 1:size(MESSAGES, 1)
        if k ~= j
            tempf = FactorProduct(tempf, lastMESSAGES(k, i));
        end
    end
    MESSAGES(i, j) = FactorMarginalization(tempf, setdiff(tempf.var, sepset));
    MESSAGES(i, j).val = MESSAGES(i, j).val ./ sum(MESSAGES(i, j).val); % normalize

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if(useSmartMP==1)
      lastMESSAGES(i,j)=prevMessage;
    end

    % store message deltas
    X = [ X MessageDelta(MESSAGES(ijij(1,1), ijij(1,2)), lastMESSAGES(ijij(1,1), ijij(1,2))) ];
    Y = [ Y MessageDelta(MESSAGES(ijij(2,1), ijij(2,2)), lastMESSAGES(ijij(2,1), ijij(2,2))) ];
    Z = [ Z MessageDelta(MESSAGES(ijij(3,1), ijij(3,2)), lastMESSAGES(ijij(3,1), ijij(3,2))) ];

    % Check for convergence every m iterations
    if mod(iteration, length(edgeFromIndx)) == 0
        if (CheckConvergence(MESSAGES, lastMESSAGES))
            break;
        end
        disp(['LBP Messages Passed: ', int2str(iteration), '...']);
        if(useSmartMP~=1)
            lastMESSAGES=MESSAGES;
        end
    end
    
end;
toc;
disp(['Total number of messages passed: ', num2str(iteration)]);

% plot message deltas
I = 1:iteration;
plot(I, X, 'r');
hold on;
plot(I, Y, 'g');
hold on;
plot(I, Z, 'b');


% Compute final potentials and place them in P
for m = 1:length(edgeFromIndx),
    j = edgeFromIndx(m);
    i = edgeToIndx(m);
    P.clusterList(i) = FactorProduct(P.clusterList(i), MESSAGES(j, i));
end


% Get the max difference between the marginal entries of 2 messages -------
function delta = MessageDelta(Mes1, Mes2)
delta = max(abs(Mes1.val - Mes2.val));
return;



% CS228 PA9 Winter 2011-2012
% File: EM_HMM.m
% Copyright (C) 2012, Stanford University

function [P loglikelihood ClassProb PairProb] = EM_HMM(actionData, poseData, G, InitialClassProb, InitialPairProb, maxIter)

% INPUTS
% actionData: structure holding the actions as described in the PA
% poseData: N x 10 x 3 matrix, where N is number of poses in all actions
% G: graph parameterization as explained in PA description
% InitialClassProb: N x K matrix, initial allocation of the N poses to the K
%   states. InitialClassProb(i,j) is the probability that example i belongs
%   to state j.
%   This is described in more detail in the PA.
% InitialPairProb: V x K^2 matrix, where V is the total number of pose
%   transitions in all HMM action models, and K is the number of states.
%   This is described in more detail in the PA.
% maxIter: max number of iterations to run EM

% OUTPUTS
% P: structure holding the learned parameters as described in the PA
% loglikelihood: #(iterations run) x 1 vector of loglikelihoods stored for
%   each iteration
% ClassProb: N x K matrix of the conditional class probability of the N examples to the
%   K states in the final iteration. ClassProb(i,j) is the probability that
%   example i belongs to state j. This is described in more detail in the PA.
% PairProb: V x K^2 matrix, where V is the total number of pose transitions
%   in all HMM action models, and K is the number of states. This is
%   described in more detail in the PA.

% Initialize variables
N = size(poseData, 1);
K = size(InitialClassProb, 2);
L = size(actionData, 2); % number of actions
V = size(InitialPairProb, 1);
numparts = size(poseData, 2);

ClassProb = InitialClassProb;
PairProb = InitialPairProb;

loglikelihood = zeros(maxIter,1);

% EM algorithm
for iter=1:maxIter
  
  % M-STEP to estimate parameters for Gaussians
  % Fill in P.c, the initial state prior probability (NOT the class probability as in PA8 and EM_cluster.m)
  % Fill in P.clg for each body part and each class
  % Make sure to choose the right parameterization based on G(i,1)
  % Hint: This part should be similar to your work from PA8 and EM_cluster.m
  
  P.c = zeros(1,K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
  
  for k=1:K
    for action=1:L
      % numposes = size(actionData(action).marg_ind, 2);
      P.c(k) = P.c(k) + ClassProb(actionData(action).marg_ind(1), k);
    end
  end
  P.c = P.c ./ sum(P.c); % normalize
  
  for part = 1:numparts

    parentpart = 0;
    U = [];
    if G(part, 1) == 1
      parentpart = G(part, 2);
      U(:, 1) = poseData(:, parentpart, 1);
      U(:, 2) = poseData(:, parentpart, 2);
      U(:, 3) = poseData(:, parentpart, 3);
    end

    for k=1:K

      if parentpart == 0
        mu_y = zeros(1, L);
        sigma_y = zeros(1, L);
        mu_x = zeros(1, L);
        sigma_x = zeros(1, L);
        mu_angle = zeros(1, L);
        sigma_angle = zeros(1, L);
      else
        theta = zeros(L, 12);
        sigma_y = zeros(1, L);
        sigma_x = zeros(1, L);
        sigma_angle = zeros(1, L);
      end

      for action=1:L

        if parentpart == 0

          [mu, sigma] = FitGaussianParameters(poseData(actionData(action).marg_ind, part, 1), ClassProb(actionData(action).marg_ind, k));
          mu_y(action) = mu;
          sigma_y(action) = sigma;

          [mu, sigma] = FitGaussianParameters(poseData(actionData(action).marg_ind, part, 2), ClassProb(actionData(action).marg_ind, k));
          mu_x(action) = mu;
          sigma_x(action) = sigma;

          [mu, sigma] = FitGaussianParameters(poseData(actionData(action).marg_ind, part, 3), ClassProb(actionData(action).marg_ind, k));
          mu_angle(action) = mu;
          sigma_angle(action) = sigma;

        else

          [Beta, sigma] = FitLinearGaussianParameters(poseData(actionData(action).marg_ind, part, 1), U(actionData(action).marg_ind, :), ClassProb(actionData(action).marg_ind, k));
          theta(action, 1) = Beta(4);
          theta(action, 2:4) = Beta(1:3);
          sigma_y(action) = sigma;

          [Beta, sigma] = FitLinearGaussianParameters(poseData(actionData(action).marg_ind, part, 2), U(actionData(action).marg_ind, :), ClassProb(actionData(action).marg_ind, k));
          theta(action, 5) = Beta(4);
          theta(action, 6:8) = Beta(1:3);
          sigma_x(action) = sigma;

          [Beta, sigma] = FitLinearGaussianParameters(poseData(actionData(action).marg_ind, part, 3), U(actionData(action).marg_ind, :), ClassProb(actionData(action).marg_ind, k));
          theta(action, 9) = Beta(4);
          theta(action, 10:12) = Beta(1:3);
          sigma_angle(action) = sigma;

        end

      end
      
      if parentpart == 0
        P.clg(part).mu_y(k) = sum( mu_y ./ sum(mu_y) );
        P.clg(part).sigma_y(k) = sum( sigma_y ./ sum(sigma_y) );
        P.clg(part).mu_x(k) = sum( mu_x ./ sum(mu_x) );
        P.clg(part).sigma_x(k) = sum( sigma_x ./ sum(sigma_x) );
        P.clg(part).mu_angle(k) = sum( mu_angle ./ sum(mu_angle) );
        P.clg(part).sigma_angle(k) = sum( sigma_angle ./ sum(sigma_angle) );
      else
        for i=1:12
          P.clg(part).theta(k, i) = sum( theta(:, i) ./ sum(theta(:, i)) );
        end
        P.clg(part).sigma_y(k) = sum( sigma_y ./ sum(sigma_y) );
        P.clg(part).sigma_x(k) = sum( sigma_x ./ sum(sigma_x) );
        P.clg(part).sigma_angle(k) = sum( sigma_angle ./ sum(sigma_angle) );
      end

    end
    
    % normalization
    if parentpart == 0
      P.clg(part).mu_y = P.clg(part).mu_y ./ sum(P.clg(part).mu_y);
      P.clg(part).sigma_y = P.clg(part).sigma_y ./ sum(P.clg(part).sigma_y);
      P.clg(part).mu_x = P.clg(part).mu_x ./ sum(P.clg(part).mu_x);
      P.clg(part).sigma_x = P.clg(part).sigma_x ./ sum(P.clg(part).sigma_x);
      P.clg(part).mu_angle = P.clg(part).mu_angle ./ sum(P.clg(part).mu_angle);
      P.clg(part).sigma_angle = P.clg(part).sigma_angle ./ sum(P.clg(part).sigma_angle);
    else
      for i=1:12
        P.clg(part).theta(:, i) = P.clg(part).theta(:, i) ./ sum(P.clg(part).theta(:, i));
      end
      P.clg(part).sigma_y = P.clg(part).sigma_y ./ sum(P.clg(part).sigma_y);
      P.clg(part).sigma_x = P.clg(part).sigma_x ./ sum(P.clg(part).sigma_x);
      P.clg(part).sigma_angle = P.clg(part).sigma_angle ./ sum(P.clg(part).sigma_angle);
    end
    

  end
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % M-STEP to estimate parameters for transition matrix
  % Fill in P.transMatrix, the transition matrix for states
  % P.transMatrix(i,j) is the probability of transitioning from state i to state j
  P.transMatrix = zeros(K,K);

  % Add Dirichlet prior based on size of poseData to avoid 0 probabilities
  P.transMatrix = P.transMatrix + size(PairProb,1) * .05;
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  Mstat = zeros(K,K);
  
  for action = 1:L
      for edge = actionData(action).pair_ind
          tempTran = reshape(PairProb(edge,:),K,K);
          Mstat = Mstat + tempTran;
      end
  end
  for sink = 1:K
      Denom = sum(Mstat(:,sink));
      for source = 1:K
          P.transMatrix(source,sink) = P.transMatrix(source,sink) + Mstat(source,sink)/Denom;
      end
  end
  %Normalize
  for source = 1:K
      P.transMatrix(source,:) = P.transMatrix(source,:)/sum(P.transMatrix(source,:));
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
    
  % E-STEP preparation: compute the emission model factors (emission probabilities) in log space for each 
  % of the poses in all actions = log( P(Pose | State) )
  % Hint: This part should be similar to (but NOT the same as) your code in EM_cluster.m
  
  logEmissionProb = zeros(N,K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  for example=1:N
    for k=1:K
      
      logEmissionProb(example, k) = 0;
      
      for part=1:numparts
        
        parentpart = 0;
        parentals = [];

        if G(part, 1) == 1
          parentpart = G(part, 2);
          parent_y = poseData(example, parentpart, 1);
          parent_x = poseData(example, parentpart, 2);
          parent_alpha = poseData(example, parentpart, 3);
          parentals = [ parent_y parent_x parent_alpha ];
        end
        
        if (parentpart == 0)
          pdf_y = lognormpdf(poseData(example, part, 1), P.clg(part).mu_y(k), P.clg(part).sigma_y(k));
          pdf_x = lognormpdf(poseData(example, part, 2), P.clg(part).mu_x(k), P.clg(part).sigma_x(k));
          pdf_angle = lognormpdf(poseData(example, part, 3), P.clg(part).mu_angle(k), P.clg(part).sigma_angle(k));

          logEmissionProb(example, k) = sum( [ logEmissionProb(example, k) pdf_y pdf_x pdf_angle ] );
        else
          mu = P.clg(part).theta(k, 1) + parentals * P.clg(part).theta(k, 2:4)';
          sigma = P.clg(part).sigma_y(k);
          pdf_y = lognormpdf(poseData(example, part, 1), mu, sigma);
          
          mu = P.clg(part).theta(k, 5) + parentals * P.clg(part).theta(k, 6:8)';
          sigma = P.clg(part).sigma_x(k);
          pdf_x = lognormpdf(poseData(example, part, 2), mu, sigma);
          
          mu = P.clg(part).theta(k, 9) + parentals * P.clg(part).theta(k, 10:12)';
          sigma = P.clg(part).sigma_angle(k);
          pdf_angle = lognormpdf(poseData(example, part, 3), mu, sigma);
          
          logEmissionProb(example, k) = sum( [ logEmissionProb(example, k) pdf_y pdf_x pdf_angle ] );
        end
        
      end
    end
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
    
  % E-STEP to compute expected sufficient statistics
  % ClassProb contains the conditional class probabilities for each pose in all actions
  % PairProb contains the expected sufficient statistics for the transition CPDs (pairwise transition probabilities)
  % Also compute log likelihood of dataset for this iteration
  % You should do inference and compute everything in log space, only converting to probability space at the end
  % Hint: You should use the logsumexp() function here to do probability normalization in log space to avoid numerical issues
  
  ClassProb = zeros(N,K);
  PairProb = zeros(V,K^2);
  loglikelihood(iter) = 0;
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  for action=1:L
    
    M = length(actionData(action).marg_ind);
    factorList = repmat(struct ('var', [], 'card', [], 'val', []), 1, 2 * M );
    currentF = 1;
    
    % P(S_1)
    % factorList(currentF).var = [ actionData(action).marg_ind(1) ];
    factorList(currentF).var = 1;
    factorList(currentF).card = [ K ];
    factorList(currentF).val = log(P.c);
    assert(all(size(factorList(currentF).val) == [ 1 prod(factorList(currentF).card) ]));
    currentF = currentF + 1;
    
    % P(S_i | S_i-1)
    
    for i=2:M
      this = actionData(action).marg_ind(i);
      prev = actionData(action).marg_ind(i-1);
      % factorList(currentF).var = [ prev this ];
      factorList(currentF).var = [i i-1];
      factorList(currentF).card = [ K K ];
      factorList(currentF).val = log(P.transMatrix(:)');
      assert(all(size(factorList(currentF).val) == [ 1 prod(factorList(currentF).card) ]));
      currentF = currentF + 1;
    end
    
    % P(P_j | S_j)
    % reduced to theta(S_j)
    
    % for i=actionData(action).marg_ind
    for i = 1:length(actionData(action).marg_ind)
      factorList(currentF).var = [i];
      factorList(currentF).card = [K];
      factorList(currentF).val = logEmissionProb(actionData(action).marg_ind(i), :);
      assert(all(size(factorList(currentF).val) == [ 1 prod(factorList(currentF).card) ]));
      currentF = currentF + 1;
    end
    
    [Marginals PCalibrated] = ComputeExactMarginalsHMM(factorList);
    
    Ps = zeros(M,K);
    for i=1:M
      Ps(i,:) = Marginals(i).val;
    end
    denom = logsumexp(logEmissionProb(actionData(action).marg_ind, :) + Ps);
    ClassProb(actionData(action).marg_ind, :) = (logEmissionProb(actionData(action).marg_ind, :) + Ps) - repmat(denom,1,K);

    % pair to pose mapping
    % actionData(action).pair_ind(1)
    % actionData(action).marg_ind(1) => actionData(action).marg_ind(2)
    
    for i=1:length(actionData(action).pair_ind)
      pair_ind = actionData(action).pair_ind(i);
      fromPose = actionData(action).marg_ind(i);
      toPose = actionData(action).marg_ind(i+1);
      for j=1:length(PCalibrated.cliqueList)
        if all(ismember([fromPose toPose], PCalibrated.cliqueList(j).var))
          PairProb(pair_ind, :) = PCalibrated.cliqueList(j).val;
          break;
        end
      end
    end
  
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Print out loglikelihood
  disp(sprintf('EM iteration %d: log likelihood: %f', ...
    iter, loglikelihood(iter)));
  if exist('OCTAVE_VERSION')
    fflush(stdout);
  end
  
  % Check for overfitting by decreasing loglikelihood
  if iter > 1
    if loglikelihood(iter) < loglikelihood(iter-1)
      break;
    end
  end
  
end

% Remove iterations if we exited early
loglikelihood = loglikelihood(1:iter);

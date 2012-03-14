% CS228 PA9 Winter 2011-2012
% File: EM_cluster.m
% Copyright (C) 2012, Stanford University

function [P loglikelihood ClassProb] = EM_cluster(poseData, G, InitialClassProb, maxIter)

% INPUTS
% poseData: N x 10 x 3 matrix, where N is number of poses;
%   poseData(example,:,:) yields the 10x3 matrix for pose example.
% G: graph parameterization as explained in PA8
% InitialClassProb: N x K, initial allocation of the N poses to the K
%   classes. InitialClassProb(example,j) is the probability that example example belongs
%   to class j
% maxIter: max number of iterations to run EM

% OUTPUTS
% P: structure holding the learned parameters as described in the PA
% loglikelihood: #(iterations run) x 1 vector of loglikelihoods stored for
%   each iteration
% ClassProb: N x K, conditional class probability of the N examples to the
%   K classes in the final iteration. ClassProb(example,j) is the probability that
%   example example belongs to class j

% Initialize variables
N = size(poseData, 1);
K = size(InitialClassProb, 2);
numparts = size(poseData, 2);

ClassProb = InitialClassProb;

loglikelihood = zeros(maxIter,1);

% EM algorithm
for iter=1:maxIter
  
  % M-STEP to estimate parameters for Gaussians
  %
  % Fill in P.c with the estimates for prior class probabilities
  % Fill in P.clg for each body part and each class
  % Make sure to choose the right parameterization based on G(example,1)
  %
  % Hint: This part should be similar to your work from PA8
  
  P.c = zeros(1,K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  for k=1:K
    P.c(k) = sum(ClassProb(:,k)) / N; % normalized
  end

  for part = 1:numparts

    for k=1:K

      parentpart = 0;
      if (length(size(G)) == 2 && G(part, 1) == 1)
        parentpart = G(part, 2);
      elseif ( length(size(G)) == 3 && G(part, 1, k) == 1)
        parentpart = G(part, 2, k);
      end

      if parentpart == 0

        [mu, sigma] = FitGaussianParameters(poseData(:, part, 1), ClassProb(:, k));
        P.clg(part).mu_y(k) = mu;
        P.clg(part).sigma_y(k) = sigma;

        [mu, sigma] = FitGaussianParameters(poseData(:, part, 2), ClassProb(:, k));
        P.clg(part).mu_x(k) = mu;
        P.clg(part).sigma_x(k) = sigma;

        [mu, sigma] = FitGaussianParameters(poseData(:, part, 3), ClassProb(:, k));
        P.clg(part).mu_angle(k) = mu;
        P.clg(part).sigma_angle(k) = sigma;

      else

        U = [];
        U(:, 1) = poseData(:, parentpart, 1);
        U(:, 2) = poseData(:, parentpart, 2);
        U(:, 3) = poseData(:, parentpart, 3);

        [Beta, sigma] = FitLinearGaussianParameters(poseData(:, part, 1), U, ClassProb(:, k));
        P.clg(part).theta(k, 1) = Beta(4);
        P.clg(part).theta(k, 2:4) = Beta(1:3);
        P.clg(part).sigma_y(k) = sigma;

        [Beta, sigma] = FitLinearGaussianParameters(poseData(:, part, 2), U, ClassProb(:, k));
        P.clg(part).theta(k, 5) = Beta(4);
        P.clg(part).theta(k, 6:8) = Beta(1:3);
        P.clg(part).sigma_x(k) = sigma;

        [Beta, sigma] = FitLinearGaussianParameters(poseData(:, part, 3), U, ClassProb(:, k));
        P.clg(part).theta(k, 9) = Beta(4);
        P.clg(part).theta(k, 10:12) = Beta(1:3);
        P.clg(part).sigma_angle(k) = sigma;

      end

    end

  end

  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % E-STEP to re-estimate ClassProb using the new parameters
  %
  % Update ClassProb with the new conditional class probabilities.
  % Recall that ClassProb(example,j) is the probability that example example belongs to
  % class j.
  %
  % You should compute everything in log space, and only convert to
  % probability space at the end.
  %
  % Tip: To make things faster, try to reduce the number of calls to
  % lognormpdf, and inline the function (example.e., copy the lognormpdf code
  % into this file)
  %
  % Hint: You should use the logsumexp() function here to do
  % probability normalization in log space to avoid numerical issues
  
  ClassProb = zeros(N,K);
  JointProb = zeros(N,K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  for example=1:N
    for k=1:K
      
      JointProb(example, k) = log(P.c(k));
      
      for part=1:numparts
        
        parentpart = 0;
        if (length(size(G)) == 2 && G(part, 1) == 1)
          parentpart = G(part, 2);
        elseif ( length(size(G)) == 3 && G(part, 1, k) == 1)
          parentpart = G(part, 2, k);
        end
        
        if (parentpart == 0)
          pdf_y = lognormpdf(poseData(example, part, 1), P.clg(part).mu_y(k), P.clg(part).sigma_y(k));
          pdf_x = lognormpdf(poseData(example, part, 2), P.clg(part).mu_x(k), P.clg(part).sigma_x(k));
          pdf_angle = lognormpdf(poseData(example, part, 3), P.clg(part).mu_angle(k), P.clg(part).sigma_angle(k));

          JointProb(example, k) = sum( [ JointProb(example, k) pdf_y pdf_x pdf_angle ] );
        else
          parent_y = poseData(example, parentpart, 1);
          parent_x = poseData(example, parentpart, 2);
          parent_alpha = poseData(example, parentpart, 3);
          parentals = [ parent_y parent_x parent_alpha ];

          mu = P.clg(part).theta(k, 1) + parentals * P.clg(part).theta(k, 2:4)';
          sigma = P.clg(part).sigma_y(k);
          pdf_y = lognormpdf(poseData(example, part, 1), mu, sigma);
          
          mu = P.clg(part).theta(k, 5) + parentals * P.clg(part).theta(k, 6:8)';
          sigma = P.clg(part).sigma_x(k);
          pdf_x = lognormpdf(poseData(example, part, 2), mu, sigma);
          
          mu = P.clg(part).theta(k, 9) + parentals * P.clg(part).theta(k, 10:12)';
          sigma = P.clg(part).sigma_angle(k);
          pdf_angle = lognormpdf(poseData(example, part, 3), mu, sigma);
          
          JointProb(example, k) = sum( [ JointProb(example, k) pdf_y pdf_x pdf_angle ] );
        end
        
      end
    end
  end
  ProbSum = logsumexp(JointProb);
  CondProb = JointProb - repmat(ProbSum,1,K);
  ClassProb = exp(CondProb);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Compute log likelihood of poseData for this iteration
  % Hint: You should use the logsumexp() function here
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  loglikelihood(iter) = 0;
  
  for example=1:N

    ll = -Inf;
    for k=1:K
      lpk = log(P.c(k));
      lpoi = 0;

      for part = 1:10

        parentpart = 0;
        if (length(size(G)) == 2 && G(part,1) == 1)
          parentpart = G(part, 2);
        elseif ( length(size(G)) == 3 && G(part,1, k) == 1)
          parentpart = G(part, 2, k);
        end

        if ( parentpart == 0 )
          lpoi = lpoi + lognormpdf( poseData(example, part, 1), P.clg(part).mu_y(k), P.clg(part).sigma_y(k) );
          lpoi = lpoi + lognormpdf( poseData(example, part, 2), P.clg(part).mu_x(k), P.clg(part).sigma_x(k) );
          lpoi = lpoi + lognormpdf( poseData(example, part, 3), P.clg(part).mu_angle(k), P.clg(part).sigma_angle(k) );
        else
          mu_y = P.clg(part).theta(k, 1) + P.clg(part).theta(k, 2) * poseData(example, parentpart, 1) + P.clg(part).theta(k, 3) * poseData(example, parentpart, 2) + P.clg(part).theta(k, 4) * poseData(example, parentpart, 3);
          mu_x = P.clg(part).theta(k, 5) + P.clg(part).theta(k, 6) * poseData(example, parentpart, 1) + P.clg(part).theta(k, 7) * poseData(example, parentpart, 2) + P.clg(part).theta(k, 8) * poseData(example, parentpart, 3);
          mu_angle = P.clg(part).theta(k, 9) + P.clg(part).theta(k, 10) * poseData(example, parentpart, 1) + P.clg(part).theta(k, 11) * poseData(example, parentpart, 2) + P.clg(part).theta(k, 12) * poseData(example, parentpart, 3);
          lpoi = lpoi + lognormpdf( poseData(example, part, 1), mu_y, P.clg(part).sigma_y(k) );
          lpoi = lpoi + lognormpdf( poseData(example, part, 2), mu_x, P.clg(part).sigma_x(k) );
          lpoi = lpoi + lognormpdf( poseData(example, part, 3), mu_angle, P.clg(part).sigma_angle(k) );
        end
      end

      ll = logsumexp( [ ll (lpk + lpoi) ]);
    end

    loglikelihood(iter) = loglikelihood(iter) + ll;
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Print out loglikelihood
  disp(sprintf('EM iteration %d: log likelihood: %f', ...
    iter, loglikelihood(iter)));
  if exist('OCTAVE_VERSION')
    fflush(stdout);
  end
  
  % Check for overfitting: when loglikelihood decreases
  if iter > 1
    if loglikelihood(iter) < loglikelihood(iter-1)
      break;
    end
  end
  
end

% Remove iterations if we exited early
loglikelihood = loglikelihood(1:iter);

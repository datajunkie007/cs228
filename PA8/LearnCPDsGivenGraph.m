% CS228 Winter 2011-2012
% File: LearnCPDsGivenGraph.m
% Copyright (C) 2012, Stanford University
% Huayan Wang

function [P loglikelihood] = LearnCPDsGivenGraph(dataset, G, labels)
%
% Inputs:
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% G: graph parameterization as explained in PA description
% labels: N x 2 true class labels for the examples. labels(i,j)=1 if the 
%         the ith example belongs to class j and 0 elsewhere        
%
% Outputs:
% P: struct array parameters (explained in PA description)
% loglikelihood: log-likelihood of the data (scalar)

N = size(dataset, 1);
K = size(labels, 2);
numparts = 10;

loglikelihood = 0;
P.c = zeros(1,K);

% estimate parameters
% fill in P.c, MLE for class probabilities
% fill in P.clg for each body part and each class
% choose the right parameterization based on G(i,1)
% compute the likelihood - you may want to use ComputeLogLikelihood.m
% you just implemented.
%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nums = sum(labels, 1);

P.c = nums ./ sum(nums);

P.clg = repmat( struct('mu_y', [], 'sigma_y', [], 'mu_x', [], 'sigma_x', [], 'mu_angle', [], 'sigma_angle', [], 'theta', [] ), 1, numparts);

for part = 1:numparts

  for k=1:K

    parentpart = 0;
    if (length(size(G)) == 2 && G(part, 1) == 1)
      parentpart = G(part, 2);
    elseif ( length(size(G)) == 3 && G(part, 1, k) == 1)
      parentpart = G(part, 2, k);
    end

    if parentpart == 0

      [mu, sigma] = FitGaussianParameters(dataset(labels(:,k) == 1, part, 1));
      P.clg(part).mu_y(k) = mu;
      P.clg(part).sigma_y(k) = sigma;

      [mu, sigma] = FitGaussianParameters(dataset(labels(:,k) == 1, part, 2));
      P.clg(part).mu_x(k) = mu;
      P.clg(part).sigma_x(k) = sigma;

      [mu, sigma] = FitGaussianParameters(dataset(labels(:,k) == 1, part, 3));
      P.clg(part).mu_angle(k) = mu;
      P.clg(part).sigma_angle(k) = sigma;

    else

      U(:, 1) = dataset(labels(:,k) == 1, parentpart, 1);
      U(:, 2) = dataset(labels(:,k) == 1, parentpart, 2);
      U(:, 3) = dataset(labels(:,k) == 1, parentpart, 3);

      [Beta, sigma] = FitLinearGaussianParameters(dataset(labels(:,k) == 1, part, 1), U);
      P.clg(part).theta(k, 1) = Beta(4);
      P.clg(part).theta(k, 2:4) = Beta(1:3);
      P.clg(part).sigma_y(k) = sigma;

      [Beta, sigma] = FitLinearGaussianParameters(dataset(labels(:,k) == 1, part, 2), U);
      P.clg(part).theta(k, 5) = Beta(4);
      P.clg(part).theta(k, 6:8) = Beta(1:3);
      P.clg(part).sigma_x(k) = sigma;

      [Beta, sigma] = FitLinearGaussianParameters(dataset(labels(:,k) == 1, part, 3), U);
      P.clg(part).theta(k, 9) = Beta(4);
      P.clg(part).theta(k, 10:12) = Beta(1:3);
      P.clg(part).sigma_angle(k) = sigma;

    end

  end

end

loglikelihood = ComputeLogLikelihood(P, G, dataset);

fprintf('log likelihood: %f\n', loglikelihood);

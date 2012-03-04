% CS228 Winter 2011-2012
% File: ComputeLogLikelihood.m
% Copyright (C) 2012, Stanford University
% Huayan Wang

function loglikelihood = ComputeLogLikelihood(P, G, dataset)
% returns the (natural) log-likelihood of data given the model and graph structure
%
% Inputs:
% P: struct array parameters (explained in PA description)
% G: graph structure and parameterization (explained in PA description)
%
%    NOTICE that G could be either 10x2 (same graph shared by all classes)
%    or 10x2x2 (each class has its own graph). your code should compute
%    the log-likelihood using the right graph.
%
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% 
% Output:
% loglikelihood: log-likelihood of the data (scalar)


N = size(dataset,1); % number of examples
K = length(P.c); % number of classes

loglikelihood = -Inf;

% You should compute the log likelihood of data as in eq. (12) and (13)
% in the PA description
% Hint: Use lognormpdf instead of log(normpdf) to prevent underflow.
%       You may use log(sum(exp(logProb))) to do addition in the original
%       space, sum(Prob).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:N

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
        lpoi = lpoi + lognormpdf( dataset(i, part, 1), P.clg(part).mu_y(k), P.clg(part).sigma_y(k) );
        lpoi = lpoi + lognormpdf( dataset(i, part, 2), P.clg(part).mu_x(k), P.clg(part).sigma_x(k) );
        lpoi = lpoi + lognormpdf( dataset(i, part, 3), P.clg(part).mu_angle(k), P.clg(part).sigma_angle(k) );
      else
        mu_y = P.clg(part).theta(k, 1) + P.clg(part).theta(k, 2) * dataset(i, parentpart, 1) + P.clg(part).theta(k, 3) * dataset(i, parentpart, 2) + P.clg(part).theta(k, 4) * dataset(i, parentpart, 3);
        mu_x = P.clg(part).theta(k, 5) + P.clg(part).theta(k, 6) * dataset(i, parentpart, 1) + P.clg(part).theta(k, 7) * dataset(i, parentpart, 2) + P.clg(part).theta(k, 8) * dataset(i, parentpart, 3);
        mu_angle = P.clg(part).theta(k, 9) + P.clg(part).theta(k, 10) * dataset(i, parentpart, 1) + P.clg(part).theta(k, 11) * dataset(i, parentpart, 2) + P.clg(part).theta(k, 12) * dataset(i, parentpart, 3);
        lpoi = lpoi + lognormpdf( dataset(i, part, 1), mu_y, P.clg(part).sigma_y(k) );
        lpoi = lpoi + lognormpdf( dataset(i, part, 2), mu_x, P.clg(part).sigma_x(k) );
        lpoi = lpoi + lognormpdf( dataset(i, part, 3), mu_angle, P.clg(part).sigma_angle(k) );
      end
    end

    loglikelihood = log( exp(loglikelihood) + exp(lpk + lpoi) );
  end

end

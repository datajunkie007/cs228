% CS228 Winter 2011-2012
% File: ClassifyDataset.m
% Copyright (C) 2012, Stanford University

function accuracy = ClassifyDataset(dataset, labels, P, G)
% returns the accuracy of the model P and graph G on the dataset 
%
% Inputs:
% dataset: N x 10 x 3, N test instances represented by 10 parts
% labels:  N x 2 true class labels for the instances.
%          labels(i,j)=1 if the ith instance belongs to class j 
% P: struct array model parameters (explained in PA description)
% G: graph structure and parameterization (explained in PA description) 
%
% Outputs:
% accuracy: percentage of correctly classified instances (scalar)

N = size(dataset, 1);
K = size(labels, 2);
accuracy = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

predictions = [];

for i=1:N

  ll = [];

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

    ll(k) = lpk + lpoi;
  end

  [maxv predictions(i)] = max(ll);
end

correct = 0;
for i = 1:N
  if labels(i, predictions(i))
    correct = correct + 1;
  end
end

accuracy = correct / N;

fprintf('Accuracy: %.2f\n', accuracy);

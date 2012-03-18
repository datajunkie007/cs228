% You should put all your code for recognizing unknown actions in this file.
% Describe the method you used in YourMethod.txt.
% Don't forget to call SavePrediction() at the end with your predicted labels to save them for submission, then submit using submit.m

function [ predicted_labels ] = RecognizeUnknownActions(G, maxIter)

load PA9Data;

datasetTrain = datasetTrain3;
datasetTest = datasetTest3;

% INPUTS
% G: graph parameterization as explained in PA decription
% maxIter: max number of iterations to run for EM

% OUTPUTS
% predicted_labels: N x 1 vector with the predicted labels for each of the instances in datasetTest, with N being the number of unknown test instances


% Train a model for each action
% Note that all actions share the same graph parameterization and number of max iterations

Ps = {};
loglikelihood = {};
ClassProbs = {};
PairProbs = {};

for action=1:length(datasetTrain)
  poseData = datasetTrain(action).poseData;
  InitialClassProb = datasetTrain(action).InitialClassProb;
  K = size(InitialClassProb, 2);
  % clustering for initial probabilities
  reshapedData = zeros(size(poseData, 1), size(poseData, 2) * size(poseData, 3));
  for pose=1:size(poseData, 1)
    reshapedData(pose) = reshape(poseData(pose, :, :), 1, size(poseData, 2) * size(poseData, 3));
  end
  gm = gmdistribution.fit(reshapedData, K);
  InitialClassProb = posterior(gm, reshapedData);
  % END clustering
  [pa ll cc pp] = EM_HMM(datasetTrain(action).actionData, poseData, G, InitialClassProb, datasetTrain(action).InitialPairProb, maxIter);
  Ps{action} = pa;
  loglikelihood{action} = ll;
  ClassProb{action} = cc;
  PairProb{action} = pp;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Classify each of the instances in datasetTrain
% Compute and return the predicted labels
% Accuracy is defined as (#correctly classified examples / #total examples)
% Note that all actions share the same graph parameterization

predicted_labels = [];

for testcase=1:length(datasetTest.actionData)

  testll = zeros(1, length(datasetTest));

  for action = 1:length(datasetTrain)

    poseData = datasetTest.poseData(datasetTest.actionData(testcase).marg_ind, :, :);
    P = Ps{action};
    actionData = datasetTest.actionData(testcase);

    N = length(actionData.marg_ind);
    K = size(ClassProb{action}, 2);
    numparts = size(datasetTest.poseData, 2);

    logEmissionProb = zeros(N, K);

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

    factorList = repmat(struct ('var', [], 'card', [], 'val', []), 1, 2 * N );
    currentF = 1;

    % P(S_1)
    % factorList(currentF).var = [ actionData(action).marg_ind(1) ];
    factorList(currentF).var = 1;
    factorList(currentF).card = [ K ];
    factorList(currentF).val = log(P.c);
    assert(all(size(factorList(currentF).val) == [ 1 prod(factorList(currentF).card) ]));
    currentF = currentF + 1;

    % P(S_i | S_i-1)

    for i=2:N
      this = actionData.marg_ind(i);
      prev = actionData.marg_ind(i-1);
      % factorList(currentF).var = [ prev this ];
      factorList(currentF).var = [i-1 i];
      factorList(currentF).card = [ K K ];
      factorList(currentF).val = log(P.transMatrix(:)');
      assert(all(size(factorList(currentF).val) == [ 1 prod(factorList(currentF).card) ]));
      currentF = currentF + 1;
    end

    % P(P_j | S_j)
    % reduced to theta(S_j)

    for i = 1:N
      factorList(currentF).var = [i];
      factorList(currentF).card = [K];
      factorList(currentF).val = logEmissionProb(i, :);
      assert(all(size(factorList(currentF).val) == [ 1 prod(factorList(currentF).card) ]));
      currentF = currentF + 1;
    end

    [Marginals PCalibrated] = ComputeExactMarginalsHMM(factorList);

    testll(action) = logsumexp(PCalibrated.cliqueList(end).val);

  end

  [ dummy predicted_labels(testcase) ] = max(testll);

end

predicted_labels = predicted_labels';
SavePredictions(predicted_labels);

% accuracy = sum(predicted_labels == datasetTest.labels) / length(datasetTest.labels)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


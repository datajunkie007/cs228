% ExtraCredit.m
% If you choose to do the extra credit portion of this assignment, you
% should do two things:
% 1) Call the function SaveECPredictions(yourPredictions).
%      `yourPredictions' should be an 80x3 matrix where
%      yourPredictions(i,j) is the predicted value (1-26) of the j'th
%      character in the i'th word. That is, the predicted value for
%      testData(i).y(j).
% 2) Copy and paste the contents of any new files you have written into
%    this file. The `submit' script needs to know the filenames in advance
%    when we collect your code, but we want to allow you to structure your
%    code any way you like. By copying those new files into this file, we
%    will collect this file and have your code.

% Copy function definitions (or just command scripts) here:
% thetaOpt = LRTrainSGD(X, y, lambda) trains a logistic regression
% classifier using stochastic gradient descent. It returns the optimal theta values. 
%
% Inputs:
% X         data.                           (numInstances x numFeatures matrix)
%           X(:,1) is all ones, i.e., it encodes the intercept/bias term.
% y         data labels.                    (numInstances x 1 vector)
% lambda    (L2) regularization parameter.  (scalar)
%
% Outputs:
% thetaOpt  optimal LR parameters.          (numFeatures x 1 vector)

load('Part2FullDataset');
load('Part2Sample');
NumParams = 2366;
modelParams = sampleModelParams;
startThetas = zeros(1,NumParams);

% This sets up an anonymous function gradFn
% such that gradFn(theta, i) = LRCostSGD(X, y, theta, lambda, i).
% We need to do this because GradientDescent takes in a function
% handle gradFunc(theta, i), where gradFunc only takes two input params.
%
% For more info, you may check out the official documentation:
% Matlab - http://www.mathworks.com/help/techdoc/matlab_prog/f4-70115.html
% Octave - http://www.gnu.org/software/octave/doc/interpreter/Anonymous-Functions.html

% totalFeatureSet = cell(length(trainData),1);
% for i = 1:length(trainData)
%     totalFeatureSet{i} = GenerateAllFeatures(trainData(i).X,trainData(i).y,modelParams);
% end
for iter = 1:2
    for i = 1:length(trainData);
    [num2str(iter) ' ' num2str(i)]
    X = trainData(i).X;
    y = trainData(i).y;
    gradFn = @(theta, i)InstanceNegLogLikelihood(X, y, theta, modelParams);

    % Calculate optimal theta values
    thetaOpt = StochasticGradientDescent(gradFn, startThetas, 1);
    startThetas = thetaOpt;
    end
end

% %%%Validation
% numTest = 50;
% numSubTest = 3;
% yourPredictions = zeros(numTest,3);
% Indecies = 1:(modelParams.numHiddenStates)^3;
% assignMatrix = IndexToAssignment(Indecies,[26,26,26]);
% for i = 1:numTest
%     i
%     valueMatrix = zeros(1,26^3);
%     featureSet = GenerateAllFeatures(trainData(i).X,[0 0 0],modelParams);
% %%%MLE
%     tempValue = 0;
%     for j = 1:length(featureSet.features)
%         if length(featureSet.features(j).var) == 1
%             tempIndex = find(assignMatrix(:,featureSet.features(j).var)==featureSet.features(j).assignment);
%         else
%             tempIndex1 = find(assignMatrix(:,featureSet.features(j).var(1))==featureSet.features(j).assignment(1));
%             tempIndex2 = find(assignMatrix(:,featureSet.features(j).var(2))==featureSet.features(j).assignment(2));
%             tempIndex = intersect(tempIndex1,tempIndex2);
%         end
%         valueMatrix(tempIndex) = valueMatrix(tempIndex) + thetaOpt(featureSet.features(j).paramIdx);
%     end
%     [maxValue,maxIdx] = max(valueMatrix);
%     yourPredictions(i,:) = IndexToAssignment(maxIdx,[26,26,26]);
% end


%%%Test
numTest = 80;
numSubTest = 3;
yourPredictions = zeros(numTest,3);
Indecies = 1:(modelParams.numHiddenStates)^3;
assignMatrix = IndexToAssignment(Indecies,[26,26,26]);
for i = 1:numTest
    i
    valueMatrix = zeros(1,26^3);
    featureSet = GenerateAllFeatures(testData(i).X,[0 0 0],modelParams);
%%%MLE
    tempValue = 0;
    for j = 1:length(featureSet.features)
        if length(featureSet.features(j).var) == 1
            tempIndex = find(assignMatrix(:,featureSet.features(j).var)==featureSet.features(j).assignment);
        else
            tempIndex1 = find(assignMatrix(:,featureSet.features(j).var(1))==featureSet.features(j).assignment(1));
            tempIndex2 = find(assignMatrix(:,featureSet.features(j).var(2))==featureSet.features(j).assignment(2));
            tempIndex = intersect(tempIndex1,tempIndex2);
        end
        valueMatrix(tempIndex) = valueMatrix(tempIndex) + thetaOpt(featureSet.features(j).paramIdx);
    end
    [maxValue,maxIdx] = max(valueMatrix);
    yourPredictions(i,:) = IndexToAssignment(maxIdx,[26,26,26]);
end



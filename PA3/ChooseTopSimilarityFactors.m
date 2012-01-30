function factors = ChooseTopSimilarityFactors (allFactors, F)
% This function chooses the similarity factors with the highest similarity
% out of all the possibilities.
%
% Input:
%   allFactors: An array of all the similarity factors.
%   F: The number of factors to select.
%
% Output:
%   factors: The F factors out of allFactors for which the similarity score
%     is highest.
%
% Hint: Recall that the similarity score for two images will be in every
%   factor table entry (for those two images' factor) where they are
%   assigned the same character value.


% If there are fewer than F factors total, just return all of them.
if (length(allFactors) <= F)
    factors = allFactors;
    return;
end

% Your code here:
% factors = allFactors; %%% REMOVE THIS LINE

scores = zeros(length(allFactors), 1);

for i = 1:length(allFactors)
    index = AssignmentToIndex([1, 1], allFactors(i).card);
    scores(i) = allFactors(i).val(index);
end

[sorted, ordering] = sort(scores);

sortedFactors = allFactors(ordering);

factors = sortedFactors(end:-1:end-F+1);

end


%GETNEXTCLUSTERS Takes in a cluster graph and returns the indices
%   of the nodes between which the next message should be passed.
%
%   [i j] = SmartGetNextClusters(P,Messages,oldMessages,m)
%
%   INPUT
%     P - our cluster graph
%     Messages - the current values of all messages in P
%     oldMessages - the previous values of all messages in P. Thus, 
%         oldMessages(i,j) contains the value that Messages(i,j) contained 
%         immediately before it was updated to its current value
%     m - the index of the message we are passing (ie, m=0 indicates we have
%         passed 0 messages prior to this one. m=5 means we've passed 5 messages
%
%     Implement any message passing routine that will converge in cases that the
%     naive routine would also converge.  You may also change the inputs to
%     this function, but note you may also have to change GetNextClusters.m as
%     well.


function [i j] = SmartGetNextClusters(P,Messages,oldMessages,m)

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  % Find the indices between which to pass a cluster
  % The 'find' function may be useful
  %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  [lli, llj] = find(P.edges, prod(size(P.edges)));
  numMessages = size(lli, 1);

  disp([ 'numMessages: ', int2str(numMessages), ' m: ', int2str(m) ]); % DEBUG

  maxdiff = 0;
  if (m-1) >= numMessages
    [ rows, cols ] = find(P.edges, numMessages);
    maxdiff = 0;
    for x = 1:size(rows)
      d = MessageDelta( Messages(rows(x), cols(x)), oldMessages(rows(x), cols(x)) );
      if d >= maxdiff
        maxdiff = d;
        i = rows(x);
        j = cols(x);
      end
    end
  else
    [i j] = NaiveGetNextClusters(P, m);
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  disp([ 'getnext returning (', int2str(i), ', ', int2str(j), ') diff: ', num2str(maxdiff) ]); % DEBUG

return;

% Get the max difference between the marginal entries of 2 messages -------
function delta = MessageDelta(Mes1, Mes2)
  delta = max(abs(Mes1.val - Mes2.val));
return;

function [numClust,clustIdx,data2cluster,numMembers,clusterAvg,clusterVar] = EstimateMode(dist,bandDist,k)

%**** Initialize stuff ****
% [numDim,numShapes] = size(X,[1 3]);
numShapes = size(dist,1);
numClust  = 0;
numMembers = [];
initShapeIdxs = 1:numShapes;
clustCent = [];                                                             % center of cluster
% stopThresh = 1e-3*bandDist;                                               % when mode has converged
beenVisitedFlag = zeros(1,numShapes);                                       % track if a shapes been seen already
numInitShapes = numShapes;                                                  % number of shapes to posibaly use as initilization shapes
clusterAvg = [];
clusterVar = [];


while numInitShapes
        
    numNeighbors = zeros(1,numShapes); 
    myMembers = [];                                                         % shapes that will get added to this cluster    
    thisClusterVotes = zeros(1,numShapes);                        
       
    %**** find mode ****
    for i = initShapeIdxs
        numNeighbors(i) = length(find(dist(i,initShapeIdxs) <= bandDist)) - 1;
    end
    
%     [numMaxNeighbor, modeIdx] = max(numNeighbors);   

    numMaxNeighbor = max(numNeighbors);
    modeAllIdx = find(numNeighbors == numMaxNeighbor);                      % find all possible modes 
    
     %**** find the central mode ****
    distMode2Members = [];
    for i = 1:length(modeAllIdx)
        %**** find members ****
        distToAll = dist(modeAllIdx(i),initShapeIdxs);                      % dist from mode to all shapes still active
        inIdxs = initShapeIdxs(find(distToAll <= bandDist));                % shapes within bandDist  

        %**** if number of new members is zero ****
        if length(inIdxs) <= k+1 || isempty(inIdxs)                         % if the num of neighbors <= k, then break     
            break;
        end       
        distMode2Members(i) = sum(dist(modeAllIdx(i),inIdxs));
        clear distToAll inIdxs;
    end   
    modeIdx = modeAllIdx(find(distMode2Members==min(distMode2Members)));   
%     myMode = X(:,:,modeIdx);  
    myMode = dist(modeIdx,modeIdx);   
       
    %**** find members ****
    distToAll = dist(modeIdx,initShapeIdxs);                                % dist from mode to all shapes still active
    inIdxs = initShapeIdxs(find(distToAll <= bandDist));                    % shapes within bandDist  
    
    %**** if number of new members is zero ****
    if length(inIdxs) <= k+1 || isempty(inIdxs)                             % if the num of members <= k, then break     
        break;
    end
    
    thisClusterVotes(inIdxs) = thisClusterVotes(inIdxs)+1;                  % add a vote for all the in shapes belonging to this cluster
    myMembers = [myMembers inIdxs];                                         % add any shapes within bandDist to the cluster
    numMembers = [numMembers length(myMembers)];
    beenVisitedFlag(myMembers) = 1;                                         % mark that these shapes have been visited
     
    numClust = numClust+1;                                                  % increment clusters
    clustCent(:,:,numClust) = myMode;                                       % record the mode      
    clustIdx(numClust) = modeIdx;                                           % record the mode index    
    clusterVotes(numClust,:) = thisClusterVotes;                            % store my members
    myOldMode = myMode;                                                     % save the old mode
    
    clusterAvg = [clusterAvg sum(dist(myMembers,modeIdx))/(length(myMembers)-1)]; % compute cluster average
    clusterVar = [clusterVar sum(dist(myMembers,modeIdx).^2)/(length(myMembers)-1)]; % compute cluster variance 

        
    initShapeIdxs = find(beenVisitedFlag == 0);                             % initialize with any of the shapes not yet visited
    numInitShapes = length(initShapeIdxs);                                  % number of active shapes in set
    
%     sprintf('Find mode shape %d',numClust)
     
end
sprintf('Find %d modes',numClust)

if numClust == 0
    clustIdx = 0;
    data2cluster = 0;
    numMembers = 0;
    clusterAvg = 0;
    clusterVar = 0;
    return
end


for iter = 1:5
    data2cluster = [];
    %**** Assign members to exsting clusters ****
    for i = 1:numShapes
        [~, data2cluster(i)] = min(dist(i,clustIdx));
    end    
    
    numOutlier = 0;
    outlierIdx = [];
    for i = 1:numClust  
       %**** find new mode for each cluster ****
       myMembers = find(data2cluster==i);
       distMode2Members = [];
       for j = 1:length(myMembers)
         distMode2Members(j) = sum(dist(myMembers(j),myMembers));
       end
       modeIdx = myMembers(find(distMode2Members==min(distMode2Members)));  
       clustIdx(i) = modeIdx;                                                   % record mode index

       %**** find outliers ****
       for j = 1:length(myMembers)
           if dist(modeIdx,myMembers(j)) > bandDist
               data2cluster(myMembers(j)) = 0;
               outlierIdx = [outlierIdx myMembers(j)];
               numOutlier = numOutlier +1;
           end
       end       
       if isempty(outlierIdx) == 0
           myMembers(find(ismember(myMembers',outlierIdx','rows'))) = [];       % remove this outlier from the cluster
       end       
       numMembers(i) = length(myMembers);                                       % record num of members for each cluster

       clusterAvg(iter,i) = sum(dist(myMembers,clustIdx(i)))/(length(myMembers)-1); % compute cluster average
       clusterVar(iter,i) = sum(dist(myMembers,clustIdx(i)).^2)/(length(myMembers)-1); % compute cluster variance 
    end
    numAllMode = numClust + numOutlier;
       
    FData = [];
    FLabel = [];
    for i = 1:numClust
      myMembers = find(data2cluster==i);
      tmp = dist(myMembers,myMembers);
      FData = [FData tmp(tril(ones(size(tmp)))>0)'];
      FLabel = [FLabel repmat(i,1,length(tmp(tril(ones(size(tmp)))>0)))];
    end
    [p,tbl] = anova1(FData,FLabel,'off');
    Fstat(iter) = tbl{2,5};

end


%**** Resort clusters (descending) ****
[~,newClustOrder] = sort(numMembers, 'descend');
oldClustIdx = clustIdx;
clustIdx = oldClustIdx(newClustOrder);
oldData2cluster = data2cluster;
for i = 1:numClust
    thisClustOrder = find(newClustOrder==i);
    data2cluster(find(oldData2cluster == i)) = thisClustOrder;
end



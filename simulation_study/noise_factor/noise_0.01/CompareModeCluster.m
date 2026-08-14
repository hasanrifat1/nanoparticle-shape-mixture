function [epsVSmode,maxFtatId,maxNumClustSetId] = CompareModeCluster(numIter,dist,k)

% X: dataset, dim = 2*m*n; n = #(sequences), m = #(points of one shape)
% dist: pairwise distance matrix for X
% k: difine the small modes: #(neighbors)<=k

%% Initialize Stuff
startValue = 0; 
widthValue = max(max(dist))/numIter;
% numShapes = size(X,3);
numShapes = size(dist,1);
numClass = 4;

%% Mode Estimation
for iter = 1:numIter

    clear clusterVotes distToAll inIdxs clustIdx clusterVotes;
	epsilon(iter) = startValue + (iter-1)*widthValue;
    bandDist = epsilon(iter);
    
    [numClust,clustIdx,data2cluster,numMembers,clusterAvg,clusterVar] = EstimateMode(dist,bandDist,k);
 
    
    %% **************************************
    numClustSet(iter) = numClust;
    numVisitedShapes(iter) = sum(numMembers);
%     numOutliers(iter) = numShapes - numVisitedShapes(iter);
    
    FData = [];
    FLabel = [];
    numMembers = [];
    for i = 1:numClust
      myMembers = find(data2cluster==i);
      numMembers(i) = length(myMembers);
      tmp = dist(myMembers,myMembers);
      FData = [FData tmp(tril(ones(size(tmp)))>0)'];
      FLabel = [FLabel repmat(i,1,length(tmp(tril(ones(size(tmp)))>0)))];
    end
    [p,tbl] = anova1(FData,FLabel,'off');
    Fstat(iter) = tbl{2,5};
end
Fstat(isnan(Fstat)) = 0;
% [~,maxFtatId] = max(Fstat);
FstatSmooth = smooth(epsilon,Fstat,0.05,'lowess');                          % smooth Fstat curve with the lowess method. Use a span of 10% of the total number of data points.
% FstatSmooth = smooth(epsilon,Fstat,0.1,'loess');                            
[~,maxFtatId] = max(FstatSmooth);
[maxNumClustSet, maxNumClustSetId] = max(numClustSet);                      % find the peak of the curve (epsilon vs. numClustSet) 

numOutliers = numShapes - numVisitedShapes;
numAllMode = numClustSet + numOutliers;
epsVSmode = [epsilon; numClustSet; numAllMode; numOutliers];




%*** Plot ***
figure(12345); clf; hold on;
xLimit = [min(epsilon) max(epsilon)];
xlim(xLimit); 
% ylim(yLimit);
yyaxis left;
yLimit = [0 max(numClustSet)+1];
ylim(yLimit);
plot(epsilon, numClustSet, 'LineWidth',2);
xlabel('Epsilon'); ylabel('Number of Significant Modes'); 
% plot(epsilon, Fstat,'LineWidth',2);
yyaxis right;
yLimit = [min(FstatSmooth) max(FstatSmooth)+1];
ylim(yLimit);
plot(epsilon, FstatSmooth,'LineWidth',2);
ylabel('Ratio (F Statistic)'); 
box on;
hold off;


figure(123467); clf; hold on;
xLimit = [min(epsilon) max(epsilon)];
yLimit = [0 max(numAllMode)+1];
xlim(xLimit); ylim(yLimit);
plot(epsilon, numAllMode, 'LineWidth',2);
plot(epsilon, numClustSet, 'LineWidth',2);
plot(epsilon, numOutliers, 'LineWidth',2);
legend('All Modes', 'Significant Modes', 'Outliers');
xlabel('Epsilon'); ylabel('Number of Modes'); 
box on;
hold off;


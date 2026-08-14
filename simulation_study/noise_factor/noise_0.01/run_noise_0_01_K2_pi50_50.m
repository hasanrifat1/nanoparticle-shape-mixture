%% ================================================================
% Representative simulation experiment
% N = 100
% Noise level sigma = 0.01
% Mixing proportions = [0.50, 0.50]
% Number of clusters K = 2
% Number of replications = 50
%% ================================================================

clear;
clc;
close all;

%% ================================================================
% Paths
%% ================================================================

addpath('./Toolbox');

save_folder = fullfile(pwd, 'results');

if ~exist(save_folder, 'dir')
    mkdir(save_folder);
end

%% ================================================================
% Load ground-truth parameters
%% ================================================================

load('pi_h_gt.mat');   % loads pi_h
load('mu_gt.mat');     % loads mu
load('C_gt.mat');      % loads C
load('m_gt.mat');      % loads m

% Store ground-truth quantities using explicit names
pi_gt = pi_h;
mu_gt = mu;
C_gt  = C;

%% ================================================================
% Simulation settings
%% ================================================================

K = 2;
N = 100;
sigma_noise = 0.01;

num_repetitions = 50;

% Reproducible replicate-specific seeds
baseSeed = 1000;

% EM tuning parameters
thresh = 1e-6;
max_iterations = 500;



%% ================================================================
% Storage for replicate-level results
%% ================================================================

q_cell = cell(1, num_repetitions);
T_cell = cell(1, num_repetitions);

Bias_mu_BEM_all = zeros(num_repetitions, 1);
Bias_mu_AEM_all = zeros(num_repetitions, 1);

Bias_C_BEM_all = zeros(num_repetitions, 1);
Bias_C_AEM_all = zeros(num_repetitions, 1);

del_pi_h_BEM_all = zeros(num_repetitions, K);
del_pi_h_AEM_all = zeros(num_repetitions, K);

runtime_all = zeros(num_repetitions, 1);

%% ================================================================
% Repetition loop
%% ================================================================

for rep = 1:num_repetitions

    rng(baseSeed + rep, 'twister');

    fprintf('\n========================================\n');
    fprintf('Running repetition %d of %d\n', rep, num_repetitions);
    fprintf('========================================\n');

    % Clear replicate-specific variables
    clear q k T T_gt resp resp_BEM resp_AEM
    clear mu1 mu2 mu1_BEM mu2_BEM
    clear C1 C2 X1 X2 x1 x2
    clear sortIdx loglike

    start_time = tic;

    %% ============================================================
    % 1. Generate synthetic shapes
    %% ============================================================

    pd = makedist( ...
        'Multinomial', ...
        'Probabilities', pi_gt);

    q = zeros(size(mu_gt{1},1), size(mu_gt{1},2), N);
    k = zeros(N,1);

    for i = 1:N

        % Generate true component label
        k(i) = random(pd);

        % Generate tangent vector from the true covariance
        x_gen = mvnrnd( ...
            zeros(2*m{k(i)},1), ...
            C_gt{k(i)} );

        x_gen = reshape(x_gen, size(mu_gt{k(i)}));

        % Map tangent vector to shape space
        q(:,:,i) = ElasticShooting( ...
            mu_gt{k(i)}, ...
            x_gen );

        % Optional coordinate-level noise:
        Original = q_to_curve(q(:,:,i));
        Original = Original + sigma_noise * randn(size(Original));
        q(:,:,i) = curve_to_q(Original);

    end

    % True labels for this replication
    T_gt = k;

    q_cell{rep} = q;
    T_cell{rep} = T_gt;

    %% ============================================================
    % 2. Compute elastic pairwise distance matrix
    %% ============================================================

    D = zeros(N,N);

    for i = 1:N
        for j = i+1:N

            q1 = squeeze(q(:,:,i));
            q2 = squeeze(q(:,:,j));

            D(i,j) = ElasticShapeDistance(q1,q2,1,0);
            D(j,i) = D(i,j);

        end
    end

    % Optional replicate-specific distance matrix
    save( ...
        fullfile(save_folder, ...
        sprintf('D_rep_%03d.mat',rep)), ...
        'D');

    %% ============================================================
    % 3. K-mode initialization
    %% ============================================================

    dist = D;
    numShapes = size(dist,1);

    % Search over bandwidth values
    numIter = 1000;

    % Small-mode/outlier threshold
    k_neighbor = ceil(numShapes * 0.05);

    [epsVSmode, maxFtatId, maxNumClustSetId] = ...
        CompareModeCluster( ...
        numIter, ...
        dist, ...
        k_neighbor);

    % Combine the two bandwidth criteria
    wF = 0.5;
    wM = 1 - wF;

    epsF = epsVSmode(1,maxFtatId);
    epsM = epsVSmode(1,maxNumClustSetId);

    bandDist = wF*epsF + wM*epsM;

    [numClust, ...
     clustIdx, ...
     data2cluster, ...
     numMembers, ...
     clusterAvg, ...
     clusterVar] = ...
        EstimateMode( ...
        dist, ...
        bandDist, ...
        k_neighbor);

    %% ============================================================
    % 4. Reassign extra clusters and outliers to two target clusters
    %% ============================================================

    if numClust < 2
        error( ...
            ['K-mode initialization produced fewer than two ', ...
             'clusters in repetition %d.'], ...
             rep);
    end

    data2clusterNew = data2cluster;

    % Initial two mode indices
    modeIdx_C1 = clustIdx(1);
    modeIdx_C2 = clustIdx(2);

    % Initial assignments
    T = data2clusterNew';

    % All observations not assigned to cluster 1 or cluster 2
    reassign_idx = find( ...
        data2clusterNew ~= 1 & ...
        data2clusterNew ~= 2);

    for ii = reassign_idx

        distToMode_C1 = ElasticShapeDistance( ...
            q(:,:,ii), ...
            q(:,:,modeIdx_C1), ...
            1,0);

        distToMode_C2 = ElasticShapeDistance( ...
            q(:,:,ii), ...
            q(:,:,modeIdx_C2), ...
            1,0);

        if distToMode_C1 < distToMode_C2
            T(ii) = 1;
        else
            T(ii) = 2;
        end

    end

    %% ============================================================
    % 5. Baseline estimates before EM (BEM)
    %% ============================================================

    mu1 = q(:,:,modeIdx_C1);
    mu2 = q(:,:,modeIdx_C2);

    resp = double(T == 1);

    resp_BEM = resp;

    mu1_BEM = mu1;
    mu2_BEM = mu2;

    [x1, X1, C1] = updateCov(mu1,q,resp);
    [x2, X2, C2] = updateCov(mu2,q,1-resp);

    w = [ ...
        sum(T == 1), ...
        sum(T == 2) ];

    w = w / sum(w);

    w_BEM = w;

    %% ============================================================
    % 6. EM algorithm
    %% ============================================================

    niter = 1;

    loglike = zeros(max_iterations + 1,1);

    loglike(niter) = ...
        loglikelihood( ...
        x1,x2,C1,C2,w);

    while niter <= max_iterations

        % ---------------------------------------------------------
        % E-step
        % ---------------------------------------------------------

        resp = updateResp( ...
            x1,x2,C1,C2,w);

        % ---------------------------------------------------------
        % M-step: mixture weights
        % ---------------------------------------------------------

        w = [ ...
            mean(resp), ...
            1-mean(resp) ];

        % ---------------------------------------------------------
        % M-step: iterative small-step mean updates
        % ---------------------------------------------------------

        mu1 = updateMean( ...
            q, ...
            mu1, ...
            resp);

        mu2 = updateMean( ...
            q, ...
            mu2, ...
            1-resp);

        % ---------------------------------------------------------
        % M-step: tangent covariance and PCA coordinates
        % ---------------------------------------------------------

        [x1, X1, C1] = ...
            updateCov(mu1,q,resp);

        [x2, X2, C2] = ...
            updateCov(mu2,q,1-resp);

        % ---------------------------------------------------------
        % Observed-data log-likelihood
        % ---------------------------------------------------------

        niter = niter + 1;

        loglike(niter) = ...
            loglikelihood( ...
            x1,x2,C1,C2,w);

        fprintf( ...
            'EM iteration %d: log-likelihood = %.10f\n', ...
            niter-1, ...
            loglike(niter));

        % ---------------------------------------------------------
        % Convergence check
        % ---------------------------------------------------------

        if abs( ...
                loglike(niter) - ...
                loglike(niter-1) ) < thresh
            break;
        end

        if niter > max_iterations
            break;
        end

    end

    loglike = loglike(1:niter);

    resp_AEM = resp;

    runtime_all(rep) = toc(start_time);

    fprintf( ...
        'EM runtime: %.3f seconds\n', ...
        runtime_all(rep));

    %% ============================================================
    % 7. Label matching based on mean-shape distances
    %% ============================================================

    % ---------------- BEM ----------------

    D1 = ...
        ElasticShapeDistance( ...
        mu_gt{1},mu1_BEM,1,0) + ...
        ElasticShapeDistance( ...
        mu_gt{2},mu2_BEM,1,0);

    D2 = ...
        ElasticShapeDistance( ...
        mu_gt{1},mu2_BEM,1,0) + ...
        ElasticShapeDistance( ...
        mu_gt{2},mu1_BEM,1,0);

    if D1 <= D2
        BEM_identity_labels = true;
        Bias_mu_BEM = D1;
    else
        BEM_identity_labels = false;
        Bias_mu_BEM = D2;
    end

    % ---------------- AEM ----------------

    D11 = ...
        ElasticShapeDistance( ...
        mu_gt{1},mu1,1,0) + ...
        ElasticShapeDistance( ...
        mu_gt{2},mu2,1,0);

    D22 = ...
        ElasticShapeDistance( ...
        mu_gt{1},mu2,1,0) + ...
        ElasticShapeDistance( ...
        mu_gt{2},mu1,1,0);

    if D11 <= D22
        AEM_identity_labels = true;
        Bias_mu_AEM = D11;
    else
        AEM_identity_labels = false;
        Bias_mu_AEM = D22;
    end

    %% ============================================================
    % 8. BEM covariance deviation
    %% ============================================================

    n = size(q,3);

    % Full covariance around BEM component 1
    X_BEM1 = zeros(length(mu1_BEM(:)),n);

    for i = 1:n

        v = ElasticShootingVector( ...
            mu1_BEM, ...
            squeeze(q(:,:,i)), ...
            1);

        X_BEM1(:,i) = v(:);

    end

    C_BEM{1} = ...
        (X_BEM1 * ...
        diag(resp_BEM) * ...
        X_BEM1') / ...
        sum(resp_BEM);

    % Full covariance around BEM component 2
    X_BEM2 = zeros(length(mu2_BEM(:)),n);

    for i = 1:n

        v = ElasticShootingVector( ...
            mu2_BEM, ...
            squeeze(q(:,:,i)), ...
            1);

        X_BEM2(:,i) = v(:);

    end

    C_BEM{2} = ...
        (X_BEM2 * ...
        diag(1-resp_BEM) * ...
        X_BEM2') / ...
        sum(1-resp_BEM);

    % Match covariance labels using mean-shape permutation
    if BEM_identity_labels

        C_ev1 = abs( ...
            eig(C_gt{1} \ C_BEM{1}) );

        C_ev2 = abs( ...
            eig(C_gt{2} \ C_BEM{2}) );

    else

        C_ev1 = abs( ...
            eig(C_gt{1} \ C_BEM{2}) );

        C_ev2 = abs( ...
            eig(C_gt{2} \ C_BEM{1}) );

    end

    del_C1 = ...
        sqrt(sum(log(C_ev1).^2));

    del_C2 = ...
        sqrt(sum(log(C_ev2).^2));

    Bias_C_BEM = del_C1 + del_C2;

    %% ============================================================
    % 9. AEM covariance deviation
    %% ============================================================

    X_AEM1 = zeros(length(mu1(:)),n);

    for i = 1:n

        v = ElasticShootingVector( ...
            mu1, ...
            squeeze(q(:,:,i)), ...
            1);

        X_AEM1(:,i) = v(:);

    end

    C_AEM{1} = ...
        (X_AEM1 * ...
        diag(resp_AEM) * ...
        X_AEM1') / ...
        sum(resp_AEM);

    X_AEM2 = zeros(length(mu2(:)),n);

    for i = 1:n

        v = ElasticShootingVector( ...
            mu2, ...
            squeeze(q(:,:,i)), ...
            1);

        X_AEM2(:,i) = v(:);

    end

    C_AEM{2} = ...
        (X_AEM2 * ...
        diag(1-resp_AEM) * ...
        X_AEM2') / ...
        sum(1-resp_AEM);

    if AEM_identity_labels

        C_ev1 = abs( ...
            eig(C_gt{1} \ C_AEM{1}) );

        C_ev2 = abs( ...
            eig(C_gt{2} \ C_AEM{2}) );

    else

        C_ev1 = abs( ...
            eig(C_gt{1} \ C_AEM{2}) );

        C_ev2 = abs( ...
            eig(C_gt{2} \ C_AEM{1}) );

    end

    del_C1 = ...
        sqrt(sum(log(C_ev1).^2));

    del_C2 = ...
        sqrt(sum(log(C_ev2).^2));

    Bias_C_AEM = del_C1 + del_C2;

    %% ============================================================
    % 10. Mixing-proportion error
    %% ============================================================

    % Empirical mixture proportions in this generated sample
    emp_pi = [ ...
        mean(T_gt == 1), ...
        mean(T_gt == 2) ];

    % ---------------- BEM ----------------

    if BEM_identity_labels

        w_BEM_aligned = ...
            [w_BEM(1),w_BEM(2)];

    else

        w_BEM_aligned = ...
            [w_BEM(2),w_BEM(1)];

    end

    del_pi_h_BEM_real = ...
        emp_pi - w_BEM_aligned;

    % ---------------- AEM ----------------

    if AEM_identity_labels

        w_AEM_aligned = ...
            [w(1),w(2)];

    else

        w_AEM_aligned = ...
            [w(2),w(1)];

    end

    del_pi_h_AEM_real = ...
        emp_pi - w_AEM_aligned;

    %% ============================================================
    % 11. Store replicate-level results
    %% ============================================================

    Bias_mu_BEM_all(rep) = Bias_mu_BEM;
    Bias_mu_AEM_all(rep) = Bias_mu_AEM;

    Bias_C_BEM_all(rep) = Bias_C_BEM;
    Bias_C_AEM_all(rep) = Bias_C_AEM;

    del_pi_h_BEM_all(rep,:) = ...
        del_pi_h_BEM_real;

    del_pi_h_AEM_all(rep,:) = ...
        del_pi_h_AEM_real;

    %% ============================================================
    % 12. Optional convergence plot
    %% ============================================================

    figure(5);
    clf;

    plot(0:length(loglike)-1,loglike,'LineWidth',1.5);

    xlabel('EM Iteration');
    ylabel('Observed-data log-likelihood');
    grid on;

    drawnow;

end

%% ================================================================
% Save final simulation results
%% ================================================================

save( ...
    fullfile(save_folder, ...
    'simulation_results_noise_0_01_K2_pi50_50.mat'), ...
    'Bias_mu_BEM_all', ...
    'Bias_mu_AEM_all', ...
    'Bias_C_BEM_all', ...
    'Bias_C_AEM_all', ...
    'del_pi_h_BEM_all', ...
    'del_pi_h_AEM_all', ...
    'runtime_all', ...
    'q_cell', ...
    'T_cell', ...
    'baseSeed', ...
    'N', ...
    'K', ...
    'sigma_noise', ...
    'num_repetitions');

fprintf('\nSimulation completed successfully.\n');



function [x, X, C] = updateCov(mu, q, resp)
    n = size(q, 3);
    X = zeros(length(mu(:)), n);
    for i = 1:n
        v = ElasticShootingVector(mu, squeeze(q(:, :, i)), 1);
        X(:, i) = v(:);
    end
    
    X2 = (X * diag(resp) * X') / sum(resp);
    [V, D] = eig(X2);
    
    [eigvals, idx] = sort(diag(D), 'descend');

    V = V(:, idx);
    eigvals = eigvals(idx);
    
    V1 = V(:, 1:9);
    x = X' * V1; % principal scores
    
    C = diag(eigvals(1:9));
end



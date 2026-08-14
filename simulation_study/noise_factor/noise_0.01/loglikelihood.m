function loglike = loglikelihood(x1, x2, C1, C2, w)

    % Number of retained tangent-space coordinates
    r = size(x1, 2);

    % Diagonal variances
    c1 = abs(diag(C1));
    c2 = abs(diag(C2));

    % Numerical safeguard
    c1 = max(c1, eps);
    c2 = max(c2, eps);

    % Log Gaussian density for component 1
    loggauss1 = ...
        -0.5 * r * log(2*pi) ...
        -0.5 * sum(log(c1)) ...
        -0.5 * sum((x1.^2) ./ c1', 2);

    % Log Gaussian density for component 2
    loggauss2 = ...
        -0.5 * r * log(2*pi) ...
        -0.5 * sum(log(c2)) ...
        -0.5 * sum((x2.^2) ./ c2', 2);

    % Add log mixture weights
    a = log(w(1)) + loggauss1;
    b = log(w(2)) + loggauss2;

    % Stable computation of:
    % log(exp(a) + exp(b))
    m = max(a, b);
    logmix = m + log(exp(a - m) + exp(b - m));

    % Observed-data log-likelihood
    loglike = sum(logmix);

end
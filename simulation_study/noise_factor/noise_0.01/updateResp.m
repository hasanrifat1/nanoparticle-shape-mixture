function [resp] = updateResp(x1, x2, C1, C2, w)
    % C1 is always diagonal
    xd1 = x1 * diag(1 ./ sqrt(abs(diag(C1))));
    gauss1 = exp(-0.5 * sum((xd1).^2, 2)) ./ prod(sqrt(abs(diag(C1))));
    
    xd2 = x2 * diag(1 ./ sqrt(abs(diag(C2))));
    gauss2 = exp(-0.5 * sum((xd2).^2, 2)) ./ prod(sqrt(abs(diag(C2))));
    
    resp = w(1) * gauss1 ./ (w(1) * gauss1 + w(2) * gauss2);
end
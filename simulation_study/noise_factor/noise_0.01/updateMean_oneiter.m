function [mu] = updateMean_oneiter(X, mu, resp)

    vm = (X * resp) / sum(resp);
    mu = ElasticShooting(mu, reshape(vm, size(mu)));
end
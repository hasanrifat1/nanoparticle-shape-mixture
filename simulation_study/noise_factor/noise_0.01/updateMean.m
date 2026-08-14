function [mu] = updateMean(q, mu, resp)
    del = 0.9;
    n = size(q,3);
    Niter= 8;
    for iter =1:Niter
        vm = 0;    
        for i=1:n
            v = resp(i) * ElasticShootingVector(mu,squeeze(q(:,:,i)),1);
            vm = vm + v;
        end
        vm = vm/sum(resp);
        mu = ElasticShooting(mu,del*vm);
    end
end

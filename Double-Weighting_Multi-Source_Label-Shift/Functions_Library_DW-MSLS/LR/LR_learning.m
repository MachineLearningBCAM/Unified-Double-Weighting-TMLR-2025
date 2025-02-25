function Mdl = LR_learning(Mdl,x)

mu = [];
n  = size(x,1);

if strcmp(Mdl.loss,'0-1')

    v  = zeros(2^Mdl.labels-1,1);

    pset = powerset(Mdl.labels);
    for i = 1:n
        for j = 1:(2^Mdl.labels-1)
            M{i,1}(j,:) = sum(phi(Mdl,x(i,:),pset{j}),1)/size(pset{j},1);
        end
    end

    for j = 1:(2^Mdl.labels-1)
        v(j,1) = 1/size(pset{j},1);
    end
    v = repmat(v,1,n);

    cvx_begin quiet
    variable mu(size(Mdl.tau,2),1)
    minimize( 1-Mdl.tau*mu+sum(max(reshape(cell2mat(M)*mu,2^Mdl.labels-1,n)-v))/n+Mdl.lambda*abs(mu)  )
    cvx_end
end

if strcmp(Mdl.loss,'log')

        for i=1:n
            M{i} = phi(Mdl,x(i,:),(1:Mdl.labels));
        end
        cvx_begin quiet
        variable mu(size(Mdl.tau,2),1)
        minimize( -Mdl.tau*mu+phi_mu(M,mu)+Mdl.lambda*abs(mu) )
        cvx_end

end

Mdl.mu       = mu;
Mdl.min_MRC = cvx_optval;

end

function value = phi_mu(M,mu)
value=0;
for k=1:length(M)
    value = value+log_sum_exp(M{k}*mu)/length(M);
end
end
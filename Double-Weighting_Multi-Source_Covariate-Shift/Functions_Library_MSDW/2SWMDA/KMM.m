function Mdl = KMM(Mdl,x_tr,x_ts)

n       = size(x_tr,1);
t       = size(x_ts,1);
epsilon = 1-1/sqrt(n);
beta    = [];
if ~exist('Mdl.B', 'var')
    Mdl.B = 1000;
end

kappa = zeros(n,1);
K     = zeros(n,n);
for i = 1:n
    K(i,i) = 1/2;
    for j = i+1:n
        K(i,j) = exp(-norm(x_tr(i,:)-x_tr(j,:))^2/(2*Mdl.sigma^2));
    end
    for j = 1:t
        kappa(i) = kappa(i)+exp(-norm(x_tr(i,:)-x_ts(j,:))^2/(2*Mdl.sigma^2));
    end
end
K = K+K';

cvx_begin quiet
variable beta(n,1)
minimize( 0.5*beta'*K*beta-(n/t)*kappa'*beta )
subject to
abs(sum(beta)-n) <= n*epsilon;
beta             >= zeros(n,1);
beta             <= Mdl.B*ones(n,1);
cvx_end

beta(beta < 0) = 0;

Mdl.beta    = beta;
Mdl.min_KMM = cvx_optval;
Mdl.K = K;
end
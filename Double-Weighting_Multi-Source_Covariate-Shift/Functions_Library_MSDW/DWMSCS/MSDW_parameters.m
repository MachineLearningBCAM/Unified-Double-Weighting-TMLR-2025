function Mdl = MSDW_parameters(Mdl,psix_tr,y_tr,psix_te)

t = size(psix_te,1);
m = size(psix_te,2);
delta = 1e-6;

for d = 1:Mdl.distribs
    auxtau = [];
    n(d) = size(psix_tr{d},1);
    for i=1:n(d)
        auxtau = [auxtau;Mdl.beta{d}(i)*phi(Mdl,psix_tr{d}(i,:),y_tr{d}(i))];
    end
    Mdl.tau{d} = sum(auxtau)/n(d);
end
Mdl.tau = cell2mat(Mdl.tau);

cvx_begin quiet
variables lambda(size(Mdl.tau)) p(t,Mdl.labels)
aux = [];
for i = 1:t
    for j = 1:Mdl.labels
        aux = [aux;p(i,j)*phi_alpha(Mdl,psix_te(i,:),j,Mdl.alpha{i})];
    end
end
minimize(ones(1,length(Mdl.tau))*lambda')
subject to
Mdl.tau-lambda+delta <= sum(aux);
sum(aux)             <= Mdl.tau+lambda-delta;
zeros(size(Mdl.tau)) <= lambda;
sum(p,2)             == ones(t,1)/t;
p                    >= 0;
cvx_end

Mdl.lambda = lambda;

for i = 1:length(Mdl.lambda)
    if Mdl.lambda(i) <= 0
        Mdl.lambda(i) = 0;
    end
end

end

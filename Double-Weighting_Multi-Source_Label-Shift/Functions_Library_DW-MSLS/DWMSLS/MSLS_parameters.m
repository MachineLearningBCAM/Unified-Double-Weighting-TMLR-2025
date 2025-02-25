function clf = MSLS_parameters(clf,x_tr,y_tr,x_te)

t = size(x_te, 1);
delta = 1e-6;
lambda = [];

if clf.distribs == 1
    auxtau = [];
    n = size(x_tr,1);
    for i=1:n
        psix_tr(i, :) = fmap(clf, x_tr(i,:));
        auxtau = [auxtau;...
            clf.beta(y_tr(i)) * phi(clf, psix_tr(i,:), y_tr(i))];
    end
    clf.tau = sum(auxtau) / n;
else
    for s = 1 : clf.distribs
        auxtau = [];
        n(s) = size(x_tr{s},1);
        for i=1:n(s)
            psix_tr{s}(i, :) = fmap(clf, x_tr{s}(i,:));
            auxtau = [auxtau;...
                clf.beta{s}(y_tr{s}(i)) * phi(clf, psix_tr{s}(i,:), y_tr{s}(i))];
        end
        clf.tau{s} = sum(auxtau) / n(s);
    end
    clf.tau = cell2mat(clf.tau);
end

cvx_begin quiet
variables lambda(size(clf.tau)) p(t,clf.labels)
aux = [];
for i = 1 : t
    psix_te(i, :) = fmap(clf, x_te(i, :));
    for j = 1:clf.labels
        aux = [aux;p(i, j) * ...
            Phi_alpha_xy(clf,psix_te(i,:),j)];
    end
end
minimize(ones(1, length(clf.tau)) * lambda')
subject to
clf.tau - lambda + delta <= sum(aux);
sum(aux)                 <= clf.tau + lambda - delta;
zeros(size(clf.tau))     <= lambda;
sum(p, 2)                == ones(t, 1) / t;
p                        >= 0;
cvx_end

clf.lambda = max(lambda, zeros(size(lambda)));

mu = [];
Mtr = [];
Mte = [];

if strcmp(clf.loss,'0-1')

    v  = zeros(2^clf.labels - 1, 1);

    pset = powerset(clf.labels);
    for i = 1 : t
        for j = 1 : (2 ^ clf.labels - 1)
            Mte{i,1}(j,:) = sum(Phi_alpha_xy(clf, psix_te(i,:),pset{j}),1) / size(pset{j}, 1);
        end
    end

    for j = 1:(2 ^ clf.labels - 1)
        v(j, 1) = 1 / size(pset{j}, 1);
    end
    v = repmat(v, 1, t);

    cvx_begin quiet
    variable mu(size(clf.tau,2),1)
    minimize( -clf.tau * mu+sum(ones(1,t)+max(reshape(cell2mat(Mte) * mu, 2 ^ clf.labels - 1, t)-v)) / t + clf.lambda * abs(mu)  )
    cvx_end
end

if strcmp(clf.loss,'log')

    for i=1:t
        Mte{i} = Phi_alpha_xy(clf, psix_te(i, :), (1 : clf.labels));
    end
    cvx_begin quiet
    variable mu(size(clf.tau, 2), 1)
    minimize( -clf.tau * mu + phi_mu(Mte, mu) / t + clf.lambda * abs(mu) )
    cvx_end
end

clf.mu       = mu;
clf.min_MRC = cvx_optval;

end

function value = phi_mu(M, mu)
value = 0;
for k = 1 : length(M)
    value = value + log_sum_exp(M{k} * mu);
end
end
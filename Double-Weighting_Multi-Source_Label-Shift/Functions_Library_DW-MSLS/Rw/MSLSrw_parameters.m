function clf = MSLSrw_parameters(clf, x_tr, y_tr)

n = size(x_tr, 1);

auxtau = [];
beta_y = zeros(1, n);
for i = 1 : n
    beta_y(i) = clf.beta(y_tr(i)); 
    psix_tr(i, :) = fmap(clf, x_tr(i, :));
    auxtau = [auxtau;...
        beta_y(i) * phi(clf, psix_tr(i, :), y_tr(i))];
end
clf.tau = sum(auxtau) / n;

clf.lambda = zeros(size(clf.tau));

mu = [];
Mtr = [];

if strcmp(clf.loss, '0-1')

    v  = zeros(2 ^ clf.labels - 1, 1);

    pset = powerset(clf.labels);
    for i = 1 : n
        for j = 1 : (2 ^ clf.labels - 1)
            Mtr{i,1}(j,:) = sum(Phi_xy(clf, psix_tr(i,:),pset{j}),1) / size(pset{j}, 1);
        end
    end

    for j = 1:(2 ^ clf.labels - 1)
        v(j, 1) = 1 / size(pset{j}, 1);
    end
    v = repmat(v, 1, n);

    cvx_begin quiet
    variable mu(size(clf.tau, 2), 1)
    minimize( -clf.tau * mu ...
        + sum(beta_y .* (ones(1, n) + max(reshape(cell2mat(Mtr) * mu, 2 ^ clf.labels - 1, n)-v))) / n + clf.lambda * abs(mu)  )
    cvx_end
end

if strcmp(clf.loss, 'log')

    for i=1:n
        Mtr{i} = Phi_xy(clf, psix_tr(i, :), (1 : clf.labels));
    end
    cvx_begin quiet
    variable mu(size(clf.tau, 2), 1)
    minimize( -clf.tau * mu + phi_beta_mu(Mtr, mu, beta_y) / n + clf.lambda * abs(mu) )
    cvx_end
end

clf.mu       = mu;
clf.min_MRC = cvx_optval;

end

function value = phi_beta_mu(M, mu,beta_y)
value = 0;
for k = 1 : length(M)
    value = value + beta_y(k) * log_sum_exp(M{k} * mu);
end
end
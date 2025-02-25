function clf = MSLS_prediction(clf, x, y)

t       = size(x,1);
fails   = 0;
y_aprox = zeros(t,1);
clf.h   = zeros(clf.labels,t);

t = size(x, 1);
for i = 1 : t
    psix(i, :) = fmap(clf, x(i,:));
end

if clf.deterministic == true
    for i = 1 : t
        [~, y_aprox(i)] = max(Phi_alpha_xy(clf, psix(i, :), 1 : clf.labels) * clf.mu);
        if y_aprox(i) ~= y(i)
            fails = fails + 1;
        end
    end
else
    if strcmp(clf.loss,'0-1')
        pset    = powerset(clf.labels);
        for i = 1:t
            for j = 1:(2^clf.labels-1)
                varphi_aux(j) = sum(Phi_alpha_xy(clf, psix(i, :),pset{j}), 1) / size(pset{j}, 1)...
                    * clf.mu - 1 / size(pset{j}, 1);
            end
            varphi_mux(i) = max(varphi_aux);
            clf.h(:, i) = max(Phi_alpha_xy(clf, psix(i, :),1:clf.labels)*clf.mu...
                - varphi_mux(i), zeros(clf.labels, 1));
            if clf.h(:, i) == zeros(clf.labels,  1)
                clf.h(:, i) = ones(clf.labels,  1) / clf.labels;
            end
        end
    end
    if strcmp(clf.loss, 'log')
        for i = 1 : t
            if any(isinf(exp(Phi_alpha_xy(clf, psix(i, :), 1 : clf.labels) * clf.mu)))
                clf.h(isinf(exp(Phi_alpha_xy(clf, psix(i, :), 1 : clf.labels) * clf.mu)), i) = ...
                    1 / sum(isinf(exp(Phi_alpha_xy(clf, psix(i, :), 1 : clf.labels) * clf.mu)));
            else
                clf.h(:, i) = exp(Phi_alpha_xy(clf, psix(i, :), 1 : clf.labels) * clf.mu)...
                / sum(exp(Phi_alpha_xy(clf, psix(i, :), 1 : clf.labels) * clf.mu));
            end
            if sum(exp(Phi_alpha_xy(clf, psix(i, :), 1 : clf.labels) * clf.mu)) == 0
                clf.h(:, i) = 1 / clf.labels;
            end
        end
    end
    for i = 1 : t
        y_aprox(i) = randsample((1 : clf.labels)', 1, true, clf.h(:, i));
        if y_aprox(i) ~= y(i)
            fails = fails + 1;
        end
    end
end

clf.error = fails/t;

end
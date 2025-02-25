function clf = MSLS_KMM(clf, xTr, yTr, xTe)

B = 10;
lambda = clf.lambda_kernel;

t = size(xTe, 1);
d = size(xTe, 2);
n = zeros(1, clf.distribs);
for s = 1 : clf.distribs
    n(s) = size(xTr{s}, 1);
    if n(s) <= 200
        sigma = 0.8 * sqrt(d);
    end
    if n(s) > 200 && n(s) <= 1200
        sigma = 0.5 * sqrt(d);
    end
    if n(s) > 1200
        sigma = 0.3 * sqrt(d);
    end
    epsilon(s) = B / (4 * sqrt(n(s)));
    alpha = [];

    Kc = zeros(t, n(s));
    Kx = zeros(n(s), n(s));
    Ky = zeros(n(s), n(s));
    for i = 1 : n(s)
        Kx(i, i) = 1 / 2;
        Ky(i, :) = (yTr{s}(i) == yTr{s}');
        for j = i + 1 : n(s)
            Kx(i,j) = exp( - norm(xTr{s}(i, :) - xTr{s}(j, :)) ^ 2 / (2 * sigma ^ 2));
        end
        for j = 1 : t
            Kc(j, i) = exp( - norm(xTe(j, :) - xTr{s}(i, :)) ^ 2 / (2*sigma ^ 2));
        end
    end
    Kx = Kx + Kx';

    R = [];
    for i = 1 : clf.labels
        R = [R, yTr{s} == (i * ones(n(s), 1))];
    end

    M2 = Ky * inv(Ky + lambda * eye(n(s)));
    A = M2 * Kx * M2';
    M = ones(1, t) * Kc * M2';

    cvx_begin quiet
    variable alpha(clf.labels, 1)
    minimize( (1 / 2) * alpha' * R' * A * R * alpha ...
        - (n(s) / t) * M * R * alpha )
    subject to
    ones(1, n(s)) * R * alpha - n(s) <= n(s) * epsilon(s);
    n(s) - ones(1, n(s)) * R * alpha <= n(s) * epsilon;
    R * alpha >= zeros(n(s),1);
    R * alpha <= B * ones(n(s),1);
    cvx_end

    clf.pte_ptr{s}    = max(alpha, 0);
    min_KMM(s) = cvx_optval;
end
clf.min_KMM = min_KMM;

end
function Mdl = MSKMM(Mdl, xTr, xTe)

t = size(xTe, 1); % t is the same for all distributions


for d = 1:Mdl.distribs
    n(d) = size(xTr{d},1);
    t = size(xTe,1);
    x = [xTr{d};xTe];
    epsilon(d) = 1-1/sqrt(n(d));

    K{d,d} = zeros(n(d)+t,n(d)+t);
    for i = 1:n(d)+t
        K{d,d}(i,i) = 1/2;
        for j = i+1:n(d)+t
            K{d,d}(i,j) = exp(-norm(x(i,:)-x(j,:))^2/(2*Mdl.sigma^2));
        end
    end
    K{d,d} = K{d,d}+K{d,d}';
    K{d,d}(1:n(d),1:n(d)) = K{d,d}(1:n(d),1:n(d))/n(d)^2;
    K{d,d}(n(d)+1:n(d)+t,n(d)+1:n(d)+t) = K{d,d}(n(d)+1:n(d)+t,n(d)+1:n(d)+t)/t^2;
    K{d,d}(1:n(d),n(d)+1:n(d)+t) = -K{d,d}(1:n(d),n(d)+1:n(d)+t)/(n(d)*t);
    K{d,d}(n(d)+1:n(d)+t,1:n(d)) = -K{d,d}(n(d)+1:n(d)+t,1:n(d))/(n(d)*t);
end
nTotal = sum(n);
% Fill in empty cells with zeros
for i = 1:size(K, 1)
    for j = 1:size(K, 2)
        if isempty(K{i, j})
            K{i, j} = zeros(size(K{i,i},1),size(K{j,j},1));
        end
    end
end
K = cell2mat(K);

cvx_begin
variable weight(nTotal+Mdl.distribs*t,1) % [beta; alpha] variables

minimize(weight' * K * weight)

idx = 1;
for d = 1:Mdl.distribs
    beta{d}  = weight(idx:idx + n(d) - 1, 1);
    alpha{d} = weight(idx + n(d):idx + n(d) + t - 1, 1);

    beta{d} >= zeros(size(xTr{d}, 1), 1);
    beta{d} <= (Mdl.B / sqrt(Mdl.D)) * ones(size(xTr{d}, 1), 1);

    alpha{d} >= zeros(t, 1);
    alpha{d} <= ones(t, 1);

    idx = idx + n(d) + t;
    abs(sum(beta{d})/n(d)-sum(alpha{d})/t) <= epsilon(d);
end

if Mdl.D == 1
    alphas_x = cvx(zeros(t, Mdl.distribs));
    for j = 1:t
        for d = 1:Mdl.distribs
            alphas_x(d,j) = alpha{d}(j);
        end
        sum(alphas_x(:,j)) == 1;
    end
else
    %Constraint for sum of alphas across distributions
    alphas_x = cvx(zeros(t, Mdl.distribs));
    for j = 1:t
        for d = 1:Mdl.distribs
            alphas_x(d,j) = alpha{d}(j);
        end
        abs(sum(alphas_x(:,j))-1) <= (1-1/sqrt(Mdl.D));
    end
end
cvx_end

Mdl.min_KMM = cvx_optval;
for d = 1:Mdl.distribs
    Mdl.beta{d} = max(beta{d}, 0);
end
for j=1:t
    for d = 1:Mdl.distribs
        Mdl.alpha{j}(d) = max(alpha{d}(j),0);
    end
end
end
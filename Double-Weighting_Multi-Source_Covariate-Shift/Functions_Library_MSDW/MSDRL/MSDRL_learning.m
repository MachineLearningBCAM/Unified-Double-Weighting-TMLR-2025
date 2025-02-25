function error = MSDRL_learning(Mdl,xTr,yTr,xTe,yTe)
labels = Mdl.labels;
if Mdl.labels == 2
    yTe(yTe == 2) = -1;
    for s = 1:Mdl.distribs
        yTr{s}(yTr{s} == 2) = -1;
    end
    for s = 1:Mdl.distribs
        fMdl{s} = Mdl;
        fMdl{s}.loss = 'log';
        fMdl{s}.fmap = 'linbin';
        fMdl{s} = model_lr(xTr{s},yTr{s});

        n(s) = size(xTr{s},1);
        idxAB = randsample(n(s),n(s));
        idxA = idxAB(1:floor(n(s)/2));
        idxB = idxAB(floor(n(s)/2)+1:end);
        xTrA{s,1} = xTr{s}(idxA,:);
        yTrA{s,1} = yTr{s}(idxA);
        xTrB{s,1} = xTr{s}(idxB,:);
        yTrB{s,1} = yTr{s}(idxB);
        betaA{s,1} = PenalizedLL(xTrA{s},xTe);
        betaB{s,1} = PenalizedLL(xTrB{s},xTe);

        fMdlA{s} = Mdl;
        fMdlA{s}.loss = 'log';
        fMdlA{s}.fmap = 'linbin';
        fMdlA{s} = model_lr(xTrA{s},yTrA{s});
        fMdlB{s} = Mdl;
        fMdlB{s}.loss = 'log';
        fMdlB{s}.fmap = 'linbin';
        fMdlB{s} = model_lr(xTrB{s},yTrB{s});
    end

    DA = zeros(Mdl.distribs,Mdl.distribs);
    DB = zeros(Mdl.distribs,Mdl.distribs);
    yTe(yTe == 2) = 0;
    for s = 1:Mdl.distribs
        yTr{s}(yTr{s} == 2) = 0;
    end
    for l = 1:Mdl.distribs
        for k = 1:Mdl.distribs
            for i = 1:size(xTrB{l},1)
                DA(l,k) = DA(l,k)+(1/size(xTrB{l},1))*betaB{l}(i)*f(fMdlA{k},xTrB{l}(i,:))*(f(fMdlA{l},xTrB{l}(i,:))-yTrB{l}(i));
            end
            for i = 1:size(xTrA{l},1)
                DB(l,k) = DB(l,k)+(1/size(xTrA{l},1))*betaA{l}(i)*f(fMdlB{k},xTrA{l}(i,:))*(f(fMdlB{l},xTrA{l}(i,:))-yTrA{l}(i));
            end
        end
    end

    GTA = zeros(Mdl.distribs,Mdl.distribs);
    GTB = zeros(Mdl.distribs,Mdl.distribs);
    GA = zeros(Mdl.distribs,Mdl.distribs);
    GB = zeros(Mdl.distribs,Mdl.distribs);
    G = zeros(Mdl.distribs,Mdl.distribs);
    for l = 1:Mdl.distribs
        for k = 1:Mdl.distribs
            for j = 1:size(xTe,1)
                GTA(l,k) = GTA(l,k)+(1/size(xTe,1))*f(fMdlA{l},xTe(j,:))*f(fMdlA{k},xTe(j,:));
                GTB(l,k) = GTB(l,k)+(1/size(xTe,1))*f(fMdlB{l},xTe(j,:))*f(fMdlB{k},xTe(j,:));
            end
            GA(l,k) = GTA(l,k)-DA(l,k)-DA(k,l);
            GB(l,k) = GTB(l,k)-DB(l,k)-DB(k,l);

            G(l,k) = (GA(l,k)+GB(l,k))/2;
        end
    end

    eig_val     = eig(G);
    [eig_vec,~] = eig(G);


    eig_val = max(eig_val, 1e-6);


    G = eig_vec * diag(eig_val) * inv(eig_vec);

    cvx_begin quiet
    variable q(Mdl.distribs,1)
    minimize( q'*G*q )
    subject to
    sum(q) == 1;
    q >= zeros(Mdl.distribs,1);
    cvx_end

    for i = 1:size(xTe)
        h = 0;
        for s = 1:Mdl.distribs
            h = h + q(s)*f(fMdl{s},xTe(i,:));
        end
        yPred(i,1)=sign(h);
    end
    error = sum(yPred ~= yTe) / numel(yTe);
else
    for s = 1:Mdl.distribs

        fMdl{s} = model_lr_multiclass(xTr{s},yTr{s});

        n(s) = size(xTr{s},1);
        idxAB = randsample(n(s),n(s));
        idxA = idxAB(1:floor(n(s)/2));
        idxB = idxAB(floor(n(s)/2)+1:end);
        xTrA{s,1} = xTr{s}(idxA,:);
        yTrA{s,1} = yTr{s}(idxA);
        xTrB{s,1} = xTr{s}(idxB,:);
        yTrB{s,1} = yTr{s}(idxB);
        betaA{s,1} = PenalizedLL(xTrA{s},xTe);
        betaB{s,1} = PenalizedLL(xTrB{s},xTe);

        fMdlA{s} = model_lr_multiclass(xTrA{s},yTrA{s});
        fMdlB{s} = model_lr_multiclass(xTrB{s},yTrB{s});
    end

    DA = zeros(Mdl.distribs,Mdl.distribs);
    DB = zeros(Mdl.distribs,Mdl.distribs);
    yTe(yTe == 2) = 0;
    for s = 1:Mdl.distribs
        yTr{s}(yTr{s} == 2) = 0;
    end
    for l = 1:Mdl.distribs
        for k = 1:Mdl.distribs
            for i = 1:size(xTrB{l},1)
                DA(l,k) = DA(l,k)+(1/size(xTrB{l},1))*betaB{l}(i)*...
                    f_multiclass(fMdlA{k},xTrB{l}(i,:),labels)'*...
                    (f_multiclass(fMdlA{l},xTrB{l}(i,:),labels)-e(yTrB{l}(i),labels)');
            end
            for i = 1:size(xTrA{l},1)
                DB(l,k) = DB(l,k)+(1/size(xTrA{l},1))*betaA{l}(i)*...
                    f_multiclass(fMdlB{k},xTrA{l}(i,:),labels)'*...
                    (f_multiclass(fMdlB{l},xTrA{l}(i,:),labels)-e(yTrA{l}(i),labels)');
            end
        end
    end

    GTA = zeros(Mdl.distribs,Mdl.distribs);
    GTB = zeros(Mdl.distribs,Mdl.distribs);
    GA = zeros(Mdl.distribs,Mdl.distribs);
    GB = zeros(Mdl.distribs,Mdl.distribs);
    G = zeros(Mdl.distribs,Mdl.distribs);
    for l = 1:Mdl.distribs
        for k = 1:Mdl.distribs
            for j = 1:size(xTe,1)
                GTA(l,k) = GTA(l,k)+(1/size(xTe,1))*...
                    f_multiclass(fMdlA{l},xTe(j,:),labels)'*...
                    f_multiclass(fMdlA{k},xTe(j,:),labels);
                GTB(l,k) = GTB(l,k)+(1/size(xTe,1))*...
                    f_multiclass(fMdlB{l},xTe(j,:),labels)'*...
                    f_multiclass(fMdlB{k},xTe(j,:),labels);
            end
            GA(l,k) = GTA(l,k)-DA(l,k)-DA(k,l);
            GB(l,k) = GTB(l,k)-DB(l,k)-DB(k,l);

            G(l,k) = (GA(l,k)+GB(l,k))/2;
        end
    end

    eig_val     = eig(G);
    [eig_vec,~] = eig(G);


    eig_val = max(eig_val, 1e-6);


    G = eig_vec * diag(eig_val) * inv(eig_vec);

    cvx_begin quiet
    variable q(Mdl.distribs,1)
    minimize( q'*G*q )
    subject to
    sum(q) == 1;
    q >= zeros(Mdl.distribs,1);
    cvx_end

    for i = 1:size(xTe)
        h = zeros(labels,1);
        for s = 1:Mdl.distribs
            h = h + q(s)*f_multiclass(fMdl{s},xTe(i,:),labels);
        end
        [~,yPred(i,1)]=max(h);
    end
    error = sum(yPred ~= yTe) / numel(yTe);
end
end


function h = f(mu,x)
h = x*mu;
end

function h = f_multiclass(mu,x,labels)
for j = 1:labels
    h(j,1)=1/sum(exp(kron(e(1:labels,labels),x)*mu-ones(labels,1)*kron(e(j,labels),x)*mu));
end
end



function mu = model_lr(x, y)
[n, d] = size(x);
mu = zeros(d, 1);

cvx_begin quiet
variable mu(d)
minimize(sum(log(1 + exp(-y .* (x * mu)))))
cvx_end
end

function mu = model_lr_multiclass(x, y)

LR.lambda0 = 0;
LR.intercept = false;
LR.fmap = 'linear';
LR.loss = 'log';
LR.lambda0 = 0;
LR.labels = numel(unique(y));
LR.distribs = 1;
LR = LR_parameters(LR, x, y);
LR = LR_learning(LR, x);
mu = LR.mu;

end

function beta = PenalizedLL(xTr,xTe)

n = size(xTr,1);
t = size(xTe,1);


X = [ones(n,1),xTr;ones(t,1),xTe];
G = [zeros(n,1);ones(t,1)];
for i = 1:size(xTr,2)+1
    normX(i) = norm(X(:,i));
end
lambda = 0.1 * sqrt(log(size(xTr,2)+1)/(n+t));


cvx_begin
variable b(size(xTr,2)+1,1)
minimize(log_b(X,G,b) + lambda*normX(2:end)*abs(b(2:end))/sqrt(n+t))
cvx_end

for i=1:n
    pG1 = exp([1,xTr(i,:)]*b)/(1+exp([1,xTr(i,:)]*b));
    beta(i) = (n/t)*(pG1/(1-pG1));
end
end

function value = log_b(X,G,b)
value=0;
for k=1:size(X,1)
    value = value+(log(1+exp(X(k,:)*b))-G(k)*X(k,:)*b)/size(X,1);
end
end
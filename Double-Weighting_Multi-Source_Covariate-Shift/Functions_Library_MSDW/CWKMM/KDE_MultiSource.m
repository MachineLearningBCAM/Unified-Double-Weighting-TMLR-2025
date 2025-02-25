function ClasWeight = KDE_MultiSource(xTr,xTe)


sigma = linspace(sqrt(1/(2*pi))-0.2*sqrt(1/(2*pi)), sqrt(1/(2*pi))+0.2*sqrt(1/(2*pi)), 10);

nDistribs = numel(xTr);
t = size(xTe,1);

for s = 1:nDistribs
    n = size(xTr{s},1);
    k = 5;
    p = [];

    %% for train

    cv = cvpartition(n,'KFold',k);
    for i = 1:k
        x_in  = xTr{s}(training(cv,i));
        x_out = xTr{s}(test(cv,i));

        for j=1:length(sigma)
            ll(i,j) = 0;
            for idx1 = 1:size(x_out,1)
                p(idx1) = probability(x_out(idx1,:),x_in,sigma(j));
                ll(i,j) = ll(i,j)+log(p(idx1));
            end
            ll(i,j) = ll(i,j)/size(x_out,1);
        end
    end
    [~,idx] = max(mean(ll));
    sigma_tr(s) = sigma(idx);
end

pTr = zeros(t,nDistribs);
for i = 1:t
    for s = 1:nDistribs
    pTr(i,s) = probability(xTe(i,:),xTr{s},sigma_tr(s));
    end
end

ClasWeight = pTr./sum(pTr,2);

nanIndices = isnan(ClasWeight);

ClasWeight(nanIndices) = 1/nDistribs;

end

%% aux function

function p = probability(x,x_in,sigma)

n = size(x_in,1);
d = size(x_in,2);
for i=1:n
    p(i) = exp(-norm(x-x_in(i,:))^2/(2*sigma^2));
end
p = 1/(n*(2*pi*sigma^2)^(d/2))*sum(p);

end
function Error = CWKMM_prediction(h_d,weights_d,yTe,deterministic,labels)

D = size(h_d,1);
t = numel(yTe);
h = zeros(labels,t);
for d = 1:D
    h = h + weights_d(:,d)' .* h_d{d};
end

yPred = zeros(t,1);

if deterministic == true
    for i = 1:t
        [~,yPred(i)] = max(h(:,i));
    end
else
    for i = 1:t
        yPred(i) = randsample((1:labels)',1,true,h(:,i));
    end
end
Error = sum(yPred ~= yTe) / numel(yTe);

end


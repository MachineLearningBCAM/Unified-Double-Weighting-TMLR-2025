function Mdl = MSDW_prediction(Mdl,psix,y)

t       = size(psix,1);
fails   = 0;
pset    = powerset(Mdl.labels);
y_aprox = zeros(t,1);
Mdl.h   = zeros(Mdl.labels,t);

if strcmp(Mdl.loss,'0-1')

    for i = 1:t
        for j = 1:(2^Mdl.labels-1)
            varphi_aux(j) = (sum(phi_alpha(Mdl,psix(i,:),pset{j}',Mdl.alpha{i})*Mdl.mu)-1)/size(pset{j},1);
        end
        varphi_mux(i) = max(varphi_aux);
        Mdl.h(:,i) = max(phi_alpha(Mdl,psix(i,:),1:Mdl.labels,Mdl.alpha{i})*Mdl.mu-varphi_mux(i),zeros(Mdl.labels,1));
        if Mdl.deterministic == true
            [~,y_aprox(i)] = max(Mdl.h(:,i));
        else
            y_aprox(i) = randsample((1:Mdl.labels)',1,true,Mdl.h(:,i));
        end
        if y_aprox(i) ~= y(i)
            fails = fails+1;
        end
    end

end

if strcmp(Mdl.loss,'log')

    for i = 1:t
        [~,y_aprox(i)]=max(phi_alpha(Mdl,psix(i,:),(1:Mdl.labels),Mdl.alpha{i})*Mdl.mu);
        if y_aprox(i) ~= y(i)
            fails = fails+1;
        end
        for j=1:Mdl.labels
            Mdl.h(j,i)=1/sum(exp(phi_alpha(Mdl,psix(i,:),(1:Mdl.labels),Mdl.alpha{i})*Mdl.mu-ones(Mdl.labels,1)*phi_alpha(Mdl,psix(i,:),j,Mdl.alpha{i})*Mdl.mu));
        end
    end

end

Mdl.error = fails/t;

end
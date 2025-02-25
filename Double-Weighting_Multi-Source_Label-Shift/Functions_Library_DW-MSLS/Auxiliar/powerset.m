function [v,R] = powerset(n_clases)
%puedes poner que devuelva R también, es la matriz del conjunto de partes

R = zeros(n_clases,2^n_clases-1);
cont = 1;
for i = 1:n_clases
    aux = nchoosek(1:n_clases,i);
    for j = 1:size(aux,1)
        for k = 1:size(aux,2)
            R(aux(j,k),cont) = 1;
        end
        cont = cont+1;
    end
end

for i = 1:2^n_clases-1
    v{i} = find(R(:,i));
end

end
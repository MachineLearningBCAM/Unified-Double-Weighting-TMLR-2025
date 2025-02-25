function map = fmap(Mdl,x)

if strcmp(Mdl.fmap,'linear')
    %Linear Kernel
    map = x;
end

if strcmp(Mdl.fmap,'polinom')
    map = [x];
    for i = 2:Mdl.degree_pol
        map = [map,x.^i];
    end
end

if strcmp(Mdl.fmap,'RFF')
    %Random Feature
    random_cos = [];
    random_sin = [];
    for i = 1:size(Mdl.u,2)
        random_cos = [random_cos,cos(x*Mdl.u(:,i))];
        random_sin = [random_sin,sin(x*Mdl.u(:,i))];
    end
    map = sqrt(1/size(Mdl.u,2))*[random_cos,random_sin];
end

end
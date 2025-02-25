function map = psi(Mdl,x)

if strcmp(Mdl.fmap,'linear')
    %Linear Kernel
    map = x;
end

if strcmp(Mdl.fmap,'polinom')
    %Polinomial Kernel (degree 2)
    map = [x,x.^2];
end

if strcmp(Mdl.fmap,'polinom3')
    %Polinomial Kernel (degree 2)
    map = [x,x.^2,x.^3];
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
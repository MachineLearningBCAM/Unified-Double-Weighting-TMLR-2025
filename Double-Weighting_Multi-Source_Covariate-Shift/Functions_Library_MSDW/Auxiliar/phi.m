function map = phi(Mdl,x,y)


if strcmp(Mdl.fmap,'linear')
    %Linear Kernel
    linear = x;
    if Mdl.intercept == true
        map = kron(e(y,Mdl.labels),[1,linear]);
    end
    if Mdl.intercept == false
        map = kron(e(y,Mdl.labels),linear);
    end
end

if strcmp(Mdl.fmap,'linbin')
    %Linear for Binary
    for i = 1:length(y)
        map(i,:) = x*y(i)/2;
    end
end

if strcmp(Mdl.fmap,'polinom')
    %Polinomial Kernel (degree 2)
    pol = [x,x.^2];
    if Mdl.intercept == true
        map = kron(e(y,Mdl.labels),[1,pol]);
    end
    if Mdl.intercept == false
        map = kron(e(y,Mdl.labels),pol);
    end
end

if strcmp(Mdl.fmap,'polinom3')
    %Polinomial Kernel (degree 2)
    pol = [x,x.^2,x.^3];
    if Mdl.intercept == true
        map = kron(e(y,Mdl.labels),[1,pol]);
    end
    if Mdl.intercept == false
        map = kron(e(y,Mdl.labels),pol);
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
    random = sqrt(1/size(Mdl.u,2))*[random_cos,random_sin];
    if Mdl.intercept == true
        map = kron(e(y,Mdl.labels),[1,random]);
    end
    if Mdl.intercept == false
        map = kron(e(y,Mdl.labels),random);
    end
end

end
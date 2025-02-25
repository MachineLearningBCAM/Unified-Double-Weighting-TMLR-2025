function Mdl = LR_parameters(Mdl,x,y)

auxtau = [];
n      = size(x,1);

for i = 1:n
    auxtau = [auxtau;phi(Mdl,x(i,:),y(i))];
end

Mdl.tau    = sum(auxtau)/n;
Mdl.lambda = Mdl.lambda0*std(auxtau)/sqrt(n);

end
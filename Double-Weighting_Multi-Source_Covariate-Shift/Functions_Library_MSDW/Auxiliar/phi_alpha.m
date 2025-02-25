function map = phi_alpha(Mdl,x,y,alpha)
map = [];
for s=1:Mdl.distribs
    map{1,s} = alpha(s)*phi(Mdl,x,y);
end
map = cell2mat(map);
end
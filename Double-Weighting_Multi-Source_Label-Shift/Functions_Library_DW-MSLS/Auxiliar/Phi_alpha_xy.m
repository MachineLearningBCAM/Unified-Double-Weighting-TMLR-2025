function map = Phi_alpha_xy(Mdl,psix,y)

if Mdl.intercept == true
    psix = [1, psix];
end
if isfield(Mdl, 'distribs') && Mdl.distribs > 1
    map = [];
    for s=1 : Mdl.distribs
        map{1, s} = Mdl.alpha{s}(y) .* kron(e(y, Mdl.labels), psix);
    end
    map = cell2mat(map);
else    
    map = Mdl.alpha(y) .* kron(e(y, Mdl.labels), psix);
end
end
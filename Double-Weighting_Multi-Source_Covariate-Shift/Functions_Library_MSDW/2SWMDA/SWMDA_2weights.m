function Mdl = SWMDA_2weights(Mdl, xTe)
t = size(xTe, 1);
Mdl.W = ones(t, t);
distances = pdist2(xTe, xTe);
for i = 1:t
    [~, sorted_indices] = sort(distances(i, :));
    nearest_neighbors_indices = sorted_indices(1:10);
    Mdl.W(i, nearest_neighbors_indices) = 0;
end
Mdl.W = Mdl.W .* Mdl.W';
Mdl.D = diag(sum(Mdl.W, 2));
Mdl.Lu = Mdl.D - Mdl.W;
matrix = Mdl.H * Mdl.Lu * Mdl.H';
cvx_begin quiet
variables w(Mdl.distribs, 1)
minimize( w' * matrix * w )
subject to
w >= zeros(Mdl.distribs, 1);
sum(w) == 1;
cvx_end
Mdl.w = w;
end
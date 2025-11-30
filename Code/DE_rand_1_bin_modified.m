function [best_sol,best_f,conv] = DE_rand_1_bin_modified(fun,dim,lb,ub,opts)
% opts.NP, opts.Gmax, opts.F, opts.CR
NP = opts.NP; Gmax = opts.Gmax; F = opts.F; CR = opts.CR;
% initialize
X = repmat(lb,NP,1) + rand(NP,dim).*repmat(ub-lb,NP,1);
fitness = zeros(NP,1);
for i=1:NP
    fitness(i) = fun(X(i,:));
end
[best_f,idx] = min(fitness); best_sol = X(idx,:);
conv = zeros(Gmax,1);
for g=1:Gmax
    for i=1:NP
        % mutation: rand/1
        idxs = randperm(NP);
        idxs(idxs==i) = [];
        a = X(idxs(1),:); b = X(idxs(2),:); c = X(idxs(3),:);
        v = a + F*(b-c);
        % crossover bin
        jrand = randi(dim);
        u = X(i,:);
        for j=1:dim
            if rand <= CR || j==jrand
                u(j) = v(j);
            end
        end
        % boundary control (repair by clipping)
        u = max(u,lb); u = min(u,ub);
        fu = fun(u);
        if fu < fitness(i)
            X(i,:) = u; fitness(i) = fu;
            if fu < best_f
                best_f = fu; best_sol = u;
            end
        end
    end
    conv(g) = best_f;
    % small modification: shrink F slowly
    F = max(0.4, F*0.999);
end
end

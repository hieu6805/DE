function [best_sol,best_f,conv] = DE_jDE(fun,dim,lb,ub,opts)
% jDE: self-adaptive control parameters for F and CR
NP = opts.NP; Gmax = opts.Gmax;
% initial F and CR for each individual
tau1 = 0.1; tau2 = 0.1;
F = 0.5 + 0.3*rand(NP,1); CR = rand(NP,1);
X = repmat(lb,NP,1) + rand(NP,dim).*repmat(ub-lb,NP,1);
fitness = zeros(NP,1);
for i=1:NP, fitness(i) = fun(X(i,:)); end
[best_f,idx] = min(fitness); best_sol = X(idx,:);
conv = zeros(Gmax,1);
for g=1:Gmax
    for i=1:NP
        % adapt parameters
        if rand < tau1, Fi = 0.1 + 0.9*rand; else Fi = F(i); end
        if rand < tau2, CRi = rand; else CRi = CR(i); end
        % mutation rand/1
        idxs = randperm(NP); idxs(idxs==i)=[];
        a=X(idxs(1),:); b=X(idxs(2),:); c=X(idxs(3),:);
        v = a + Fi*(b-c);
        % crossover
        jrand = randi(dim);
        u = X(i,:);
        for j=1:dim
            if rand <= CRi || j==jrand
                u(j) = v(j);
            end
        end
        u = max(u,lb); u = min(u,ub);
        fu = fun(u);
        if fu < fitness(i)
            X(i,:) = u; fitness(i) = fu;
            F(i) = Fi; CR(i) = CRi; % accept adapted params
            if fu < best_f, best_f = fu; best_sol = u; end
        end
    end
    conv(g) = best_f;
end
end

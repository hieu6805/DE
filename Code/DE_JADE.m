function [best_sol,best_f,conv] = DE_JADE(fun,dim,lb,ub,opts)
% Simplified JADE implementation with current-to-pbest/1 and external archive
NP = opts.NP; Gmax = opts.Gmax; mu_F = 0.5; mu_CR = 0.5;
p = 0.2; % p-best rate
A = []; % archive
X = repmat(lb,NP,1) + rand(NP,dim).*repmat(ub-lb,NP,1);
fitness = zeros(NP,1); for i=1:NP, fitness(i)=fun(X(i,:)); end
[best_f,idx] = min(fitness); best_sol = X(idx,:);
conv = zeros(Gmax,1);
for g=1:Gmax
    S_F = []; S_CR = [];
    for i=1:NP
        % generate Fi and CRi from Cauchy and Normal
        Fi = cauchy_rand(mu_F,0.1); while Fi<=0, Fi=cauchy_rand(mu_F,0.1); end
        Fi = min(Fi,1);
        CRi = mu_CR + 0.1*randn; CRi = min(max(CRi,0),1);
        % choose p-best
        [~,order] = sort(fitness);
        pbest_idx = order(randi(max(1,round(p*NP))));
        % mutation current-to-pbest/1 with archive
        idxs = randperm(NP+size(A,1));
        % build combined population
        if isempty(A), Pop = X; else Pop = [X; A]; end
        % ensure we have enough distinct indices
        ids = idxs(idxs~=i); ids = ids(1:min(3,numel(ids)));
        r1 = Pop(ids(1),:);
        r2 = Pop(ids(2),:);
        v = X(i,:) + Fi*(X(pbest_idx,:)-X(i,:)) + Fi*(r1 - r2);
        % crossover
        jrand = randi(dim); u = X(i,:);
        for j=1:dim
            if rand <= CRi || j==jrand, u(j)=v(j); end
        end
        u = max(u,lb); u = min(u,ub);
        fu = fun(u);
        if fu < fitness(i)
            % success
            A = [A; X(i,:)]; if size(A,1)>NP, A(1,:)=[]; end
            X(i,:) = u; fitness(i)=fu; S_F(end+1)=Fi; S_CR(end+1)=CRi;
            if fu < best_f, best_f = fu; best_sol = u; end
        end
    end
    % update mu_F and mu_CR using Lehmer mean for F and arithmetic mean for CR
    if ~isempty(S_F)
        mu_F = (sum(S_F.^2)/sum(S_F));
    end
    if ~isempty(S_CR)
        mu_CR = mean(S_CR);
    end
    conv(g) = best_f;
end
end

function x = cauchy_rand(loc,scale)
% sample from Cauchy(loc, scale) using inverse CDF
u = rand - 0.5;
x = loc + scale * tan(pi*u);
end

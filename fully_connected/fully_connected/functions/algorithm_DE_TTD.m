function [t_opt, R_history] = algorithm_DE_TTD(para, H, A_PS)
    % Tham số thuật toán DE
    NP = 50;            
    max_gen = 100;       
    F = 0.8;            
    CR = 0.9;           
    
    % Số biến cần tối ưu là N_T * N_RF (số bộ TTD thực tế) [cite: 119]
    D = para.N_T * para.N_RF; 
    
    lb = zeros(1, D);
    ub = ones(1, D) * para.t_max;
    
    pop = lb + (ub - lb) .* rand(NP, D);
    fit = zeros(NP, 1);
    
    for i = 1:NP
        fit(i) = fitness_DE_TTD(pop(i,:), para, H, A_PS);
    end
    
    [best_fit, idx] = min(fit);
    best_sol = pop(idx, :);
    R_history = zeros(max_gen, 1);
    
    for gen = 1:max_gen
        for i = 1:NP
            % Đột biến và Lai ghép (DE/rand/1/bin)
            r = randperm(NP, 3);
            while any(r == i), r = randperm(NP, 3); end
            mutant = pop(r(1),:) + F * (pop(r(2),:) - pop(r(3),:));
            mutant = max(min(mutant, ub), lb);
            
            trial = pop(i, :);
            j_rand = randi(D);
            for j = 1:D
                if rand < CR || j == j_rand
                    trial(j) = mutant(j);
                end
            end
            
            f_trial = fitness_DE_TTD(trial, para, H, A_PS);
            if f_trial < fit(i)
                pop(i,:) = trial;
                fit(i) = f_trial;
                if f_trial < best_fit
                    best_fit = f_trial;
                    best_sol = trial;
                end
            end
        end
        R_history(gen) = -best_fit;
    end
    t_opt = reshape(best_sol, [para.N_T, para.N_RF]);
end
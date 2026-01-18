function [t_opt, R_history] = algorithm_DE_pure(para, H, A_PS)
    % Tăng cường tham số để DE đủ sức tìm kiếm
    NP = 40;            % Tăng số cá thể
    max_gen = 100;      % Tăng số thế hệ
    F = 0.6;            % Hệ số đột biến thấp hơn chút để tinh chỉnh (Fine-tuning)
    CR = 0.9;           
    D = para.N_T * para.N_RF; 
    
    lb = 0; ub = para.t_max;
    % Khởi tạo ngẫu nhiên hoàn toàn 100%
    pop = lb + (ub - lb) .* rand(NP, D);
    fit = zeros(NP, 1);
    
    fprintf('Khởi tạo quần thể DE ngẫu nhiên...\n');
    for i = 1:NP
        fit(i) = fitness_DE_pure(pop(i,:), para, H, A_PS);
    end
    
    [best_fit, idx] = min(fit);
    best_sol = pop(idx, :);
    R_history = zeros(max_gen, 1);
    
    for gen = 1:max_gen
        for i = 1:NP
            % Chiến lược DE/best/1: Đột biến quanh cá thể tốt nhất
            r = randperm(NP, 2);
            mutant = best_sol + F * (pop(r(1),:) - pop(r(2),:));
            mutant = max(min(mutant, ub), lb);
            
            % Lai ghép (Crossover)
            trial = pop(i, :);
            j_rand = randi(D);
            for j = 1:D
                if rand < CR || j == j_rand
                    trial(j) = mutant(j);
                end
            end
            
            % Lựa chọn (Selection)
            f_trial = fitness_DE_pure(trial, para, H, A_PS);
            if f_trial <= fit(i)
                pop(i,:) = trial;
                fit(i) = f_trial;
                if f_trial < best_fit
                    best_fit = f_trial;
                    best_sol = trial;
                end
            end
        end
        R_history(gen) = -best_fit;
        if mod(gen, 10) == 0
            fprintf('Thế hệ %d: SE hiện tại = %.4f\n', gen, -best_fit);
        end
    end
    t_opt = reshape(best_sol, [para.N_T, para.N_RF]);
end
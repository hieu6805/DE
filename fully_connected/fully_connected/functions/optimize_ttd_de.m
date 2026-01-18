%% --- Local Function: Differential Evolution for TTD ---
function best_sol = optimize_ttd_de(tau_prop, fm_all, t_geo, opts)
    % Inputs:
    %   tau_prop: vector (r_l - r)/c
    %   fm_all: vector of frequencies
    %   t_geo: geometric initialization guess
    %   opts: DE parameters
    
    D = length(tau_prop); % Dimension of the problem (N_T)
    NP = opts.pop_size;
    
    % 1. Initialization
    % Khởi tạo quần thể ngẫu nhiên trong khoảng [lb, ub]
    pop = opts.lb + (opts.ub - opts.lb) * rand(NP, D);
    % Inject geometric solution into population (giúp hội tụ nhanh hơn)
    pop(1, :) = t_geo'; 
    
    % Evaluate initial fitness
    fitness = zeros(NP, 1);
    for i = 1:NP
        fitness(i) = objective_function(pop(i,:)', tau_prop, fm_all);
    end
    
    % Tìm cá thể tốt nhất ban đầu
    [best_val, idx] = max(fitness);
    best_sol = pop(idx, :)';
    
    % 2. DE Main Loop
    for iter = 1:opts.max_iter
        new_pop = pop;
        for i = 1:NP
            % Mutation: select r1, r2, r3 distinct and != i
            idxs = randperm(NP);
            idxs(idxs == i) = [];
            r1 = idxs(1); r2 = idxs(2); r3 = idxs(3);
            
            % Mutation vector v
            v = pop(r1,:) + opts.F * (pop(r2,:) - pop(r3,:));
            
            % Boundary handling (Clamp)
            v = max(v, opts.lb);
            v = min(v, opts.ub);
            
            % Crossover
            u = pop(i,:);
            j_rand = randi(D);
            mask = (rand(1, D) < opts.CR);
            mask(j_rand) = true; % Ensure at least one element changes
            u(mask) = v(mask);
            
            % Selection
            fit_u = objective_function(u', tau_prop, fm_all);
            
            if fit_u > fitness(i)
                new_pop(i,:) = u;
                fitness(i) = fit_u;
                
                % Update global best
                if fit_u > best_val
                    best_val = fit_u;
                    best_sol = u';
                end
            end
        end
        pop = new_pop;
    end
end

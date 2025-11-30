function [best_fitness, best_solution, fitness_history] = de_rastrigin_jade()
% JADE (Adaptive Differential Evolution with Optional External Archive) - Optimized Version
% Optimization: Minimizing the Rastrigin function
% Improvement: Optimized selection of pbest by reusing sorted indices and periodic sorting.
% --- 1. Problem Parameter Definition ---
N_DIM = 10;     % Dimension (n)
LB = -5.12;     % Lower bound
UB = 5.12;      % Upper bound
% --- 2. JADE Algorithm Parameter Definition ---
POP_SIZE = 100;  % Population size (N)
MAX_GEN = 1000; % Maximum number of generations
P_BEST = 0.05;  % p ratio in DE/current-to-pbest/1 (p * POP_SIZE)
C = 0.1;        % Learning factor for updating mu_F and mu_CR
MAX_ARC_SIZE = POP_SIZE; % Optimized: Set max Archive size equal to Pop size (N)
SORT_INTERVAL = 10; % Optimization: Sort the population fully only every 10 generations
% Adaptive parameter initialization
mu_F = 0.5;     % Mean F for Cauchy
mu_CR = 0.5;    % Mean CR for Normal
Archive = [];   % Stores failed vectors
Fitness = zeros(POP_SIZE, 1);
fitness_history = zeros(MAX_GEN + 1, 1);
% --- 3. Population Initialization ---
Pop = LB + rand(POP_SIZE, N_DIM) .* (UB - LB);
for i = 1:POP_SIZE
   Fitness(i) = rastrigin_function(Pop(i, :));
end
[best_fitness, best_idx] = min(Fitness);
best_solution = Pop(best_idx, :);
fitness_history(1) = best_fitness;
% Optimization 1: Initial full sort to get indices (O(N log N))
[~, sorted_idx] = sort(Fitness);
fprintf('\nStarting Optimization: JADE (Optimized) | MAX_GEN=%d\n', MAX_GEN);
disp('=====================================');
disp('| Generation | Best Fitness     |');
disp('=====================================');
fprintf('| %-10d | %-16.8f |\n', 0, best_fitness);
% --- 4. Main Evolutionary Loop ---
for gen = 1:MAX_GEN
   NewPop = zeros(POP_SIZE, N_DIM);
   NewFitness = zeros(POP_SIZE, 1);
  
   S_F = [];
   S_CR = [];
  
   % Generate F_i and CR_i (Retaining Cauchy/Normal logic)
   F_vec = zeros(POP_SIZE, 1);
   CR_vec = normrnd(mu_CR, 0.1, POP_SIZE, 1);
   CR_vec(CR_vec > 1) = 1;
   CR_vec(CR_vec < 0) = 0;
   for i = 1:POP_SIZE
       while true
           F_i_raw = mu_F + 0.1 * tan(pi * (rand() - 0.5));
           if F_i_raw > 0
               F_vec(i) = min(F_i_raw, 1);
               break;
           end
       end
   end
  
   % Optimization 1: Select pbest candidates from the pre-sorted indices (O(1) lookup)
   num_pbest = max(2, round(POP_SIZE * P_BEST));
   % Get pbest candidates using the sorted indices from the last full sort
   P_best_candidates = Pop(sorted_idx(1:num_pbest), :);
  
   % --- Loop for each individual ---
   for i = 1:POP_SIZE
       Target_x = Pop(i, :);
       F_i = F_vec(i);
       CR_i = CR_vec(i);
       % A. Mutation: DE/current-to-pbest/1
      
       % 1. Select pbest randomly from the candidates (O(1))
       p_best_idx = randi(num_pbest);
       pbest = P_best_candidates(p_best_idx, :);
      
       % 2. Select r1, r2 from Unified Population (Pop + Archive)
      
       Union_Size = POP_SIZE + size(Archive, 1);
      
       % Select r1 (from Pop, distinct from i)
       indices_P = 1:POP_SIZE; indices_P(i) = [];
       r1_idx_P = indices_P(randi(length(indices_P)));
       r1 = Pop(r1_idx_P, :);
      
       % Select r2 (from Pop or Archive)
      
       % Optimization 2: Access Pop or Archive directly via index (avoid building Union_Pop array)
       r2_idx_Union = randi(Union_Size);
       if r2_idx_Union <= POP_SIZE
           % r2 selected from Pop
           r2 = Pop(r2_idx_Union, :);
       else
           % r2 selected from Archive
           r2_idx_A = r2_idx_Union - POP_SIZE;
           r2 = Archive(r2_idx_A, :);
       end
      
       % Mutated Vector V
       V = Target_x + F_i * (pbest - Target_x) + F_i * (r1 - r2);
      
       % Handle boundary violations
       V(V < LB) = LB;
       V(V > UB) = UB;
      
       % B. Crossover & C. Selection (No change)
       Trial_x = Target_x;
       j_rand = randi(N_DIM);
      
       for j = 1:N_DIM
           if rand() < CR_i || j == j_rand
               Trial_x(j) = V(j);
           end
       end
      
       TrialFitness = rastrigin_function(Trial_x);
      
       if TrialFitness < Fitness(i)
           % Success: Accept new solution and store parameters
           NewPop(i, :) = Trial_x;
           NewFitness(i) = TrialFitness;
           S_F = [S_F; F_i];
           S_CR = [S_CR; CR_i];
          
           % Add failed individual to Archive
           Archive = [Archive; Target_x];
          
       else
           % Failure: Retain old solution
           NewPop(i, :) = Pop(i, :);
           NewFitness(i) = Fitness(i);
       end
   end
  
   % D. Archive Management (Keep max size at MAX_ARC_SIZE = POP_SIZE)
   if size(Archive, 1) > MAX_ARC_SIZE
       % Randomly remove surplus individuals (O(N))
       num_to_remove = size(Archive, 1) - MAX_ARC_SIZE;
       perm_idx = randperm(size(Archive, 1));
       Archive(perm_idx(1:num_to_remove), :) = [];
   end
  
   % E. Update mu_F and mu_CR
   if ~isempty(S_F)
       mu_CR = (1 - C) * mu_CR + C * mean(S_CR);
       mu_F = (1 - C) * mu_F + C * (sum(S_F.^2) / sum(S_F));
   end
  
   % Update Population
   Pop = NewPop;
   Fitness = NewFitness;
  
   % Optimization 1 (cont.): Periodically update the sorted indices
   if mod(gen, SORT_INTERVAL) == 0 || gen == MAX_GEN
       % Full sort (O(N log N)) only every SORT_INTERVAL generations
       [~, sorted_idx] = sort(Fitness);
   end
  
   % Update global best solution
   [current_best_f, current_best_idx] = min(Fitness);
   if current_best_f < best_fitness
       best_fitness = current_best_f;
       best_solution = Pop(current_best_idx, :);
   end
  
   fitness_history(gen + 1) = best_fitness;
  
   % Print progress
   if mod(gen, 100) == 0
       fprintf('| %-10d | %-16.8f |\n', gen, best_fitness);
   end
end
% --- Print Final Results ---
disp('=====================================');
fprintf('Final Result (after %d generations): %.8f\n', MAX_GEN, best_fitness);
fprintf('Best Solution found: (');
fprintf('%.4f, ', best_solution(1:end-1));
fprintf('%.4f)\n', best_solution(end));
end

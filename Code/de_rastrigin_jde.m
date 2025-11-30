function [best_fitness, best_solution, fitness_history] = de_rastrigin_jde()
% jDE (Self-Adaptive DE) - Differential Evolution with self-adaptive parameters
% Optimization: Minimizing the Rastrigin function
% Mechanism: Automatically adjusts F and CR for each individual (DE/rand/1/bin).
% --- 1. Problem Parameter Definition ---
N_DIM = 10;     % Dimension (n)
LB = -5.12;     % Lower bound
UB = 5.12;      % Upper bound
% --- 2. jDE Algorithm Parameter Definition ---
POP_SIZE = 100;  % Population size
MAX_GEN = 1000; % Maximum number of generations
TAU1 = 0.1;     % Probability of changing F
TAU2 = 0.1;     % Probability of changing CR
F_L = 0.1;      % Lower bound for F
F_U = 1.0;      % Upper bound for F
% Pop structure: [Solution x (N_DIM) | F (1) | CR (1)]
Pop = zeros(POP_SIZE, N_DIM + 2);
Fitness = zeros(POP_SIZE, 1);
fitness_history = zeros(MAX_GEN + 1, 1);
% --- 3. Population Initialization ---
% Initialize x positions randomly
Pop(:, 1:N_DIM) = LB + rand(POP_SIZE, N_DIM) .* (UB - LB);
% Initialize initial F and CR for each individual
Pop(:, N_DIM + 1) = F_L + rand(POP_SIZE, 1) .* (F_U - F_L); % Initial F random in [0.1, 1.0]
Pop(:, N_DIM + 2) = rand(POP_SIZE, 1);                      % Initial CR random in [0, 1]
% Calculate initial Fitness
for i = 1:POP_SIZE
   x = Pop(i, 1:N_DIM);
   Fitness(i) = rastrigin_function(x);
end
% Find the global best solution
[best_fitness, best_idx] = min(Fitness);
best_solution = Pop(best_idx, 1:N_DIM);
fitness_history(1) = best_fitness;
fprintf('\nStarting Optimization: jDE (Self-Adaptive) | MAX_GEN=%d\n', MAX_GEN);
disp('=====================================');
disp('| Generation | Best Fitness |');
disp('=====================================');
fprintf('| %-10d | %-12.8f |\n', 0, best_fitness);
% --- 4. Main Evolutionary Loop ---
for gen = 1:MAX_GEN
   NewPop = zeros(POP_SIZE, N_DIM + 2);
   NewFitness = zeros(POP_SIZE, 1);
  
   for i = 1:POP_SIZE
       Target_x = Pop(i, 1:N_DIM);
       Target_F = Pop(i, N_DIM + 1);
       Target_CR = Pop(i, N_DIM + 2);
      
       % A. Update F and CR Parameters (jDE Adaptation)
       F_i = Target_F;
       CR_i = Target_CR;
      
       % Update F with probability TAU1 (select new random F)
       if rand() < TAU1
           F_i = F_L + rand() * (F_U - F_L);
       end
       % Update CR with probability TAU2 (select new random CR)
       if rand() < TAU2
           CR_i = rand();
       end
      
       % Constrain F and CR to their bounds
       F_i = min(max(F_i, F_L), F_U);
       CR_i = min(max(CR_i, 0), 1);
      
       % B. Mutation (Mutation: DE/rand/1)
       % Randomly select 3 individuals other than i: r1, r2, r3
       indices = 1:POP_SIZE; indices(i) = [];
       r = indices(randperm(POP_SIZE - 1, 3));
       r1 = r(1); r2 = r(2); r3 = r(3);
      
       % Mutated Vector V = X_r3 + F_i * (X_r2 - X_r1)
       V = Pop(r3, 1:N_DIM) + F_i * (Pop(r2, 1:N_DIM) - Pop(r1, 1:N_DIM));
       % Handle boundary violations
       V(V < LB) = LB;
       V(V > UB) = UB;
      
       % C. Binomial Crossover
       Trial_x = Target_x;
       j_rand = randi(N_DIM); % Ensures at least one component is changed
      
       for j = 1:N_DIM
           if rand() < CR_i || j == j_rand
               Trial_x(j) = V(j);
           end
       end
      
       % D. Selection
       TrialFitness = rastrigin_function(Trial_x);
      
       if TrialFitness < Fitness(i)
           % If successful, accept trial vector AND the new parameters F_i, CR_i
           NewPop(i, 1:N_DIM) = Trial_x;
           NewPop(i, N_DIM + 1) = F_i;   
           NewPop(i, N_DIM + 2) = CR_i;  
           NewFitness(i) = TrialFitness;
       else
           % If failed, retain old target vector and old parameters
           NewPop(i, :) = Pop(i, :);
           NewFitness(i) = Fitness(i);
       end
   end
  
   % Update Population (Simultaneous Update)
   Pop = NewPop;
   Fitness = NewFitness;
  
   % Update global best solution
   [current_best_f, current_best_idx] = min(Fitness);
   if current_best_f < best_fitness
       best_fitness = current_best_f;
       best_solution = Pop(current_best_idx, 1:N_DIM);
   end
  
   fitness_history(gen + 1) = best_fitness;
  
   % Print progress every 100 generations
   if mod(gen, 100) == 0
       fprintf('| %-10d | %-12.8f |\n', gen, best_fitness);
   end
end
% --- Print Final Results ---
disp('=====================================');
fprintf('Final Result (after %d generations): %.8f\n', MAX_GEN, best_fitness);
fprintf('Best Solution found: (');
fprintf('%.4f, ', best_solution(1:end-1));
fprintf('%.4f)\n', best_solution(end));
end

function [best_fitness, best_solution, fitness_history] = de_rastrigin_rand1()
% DE/rand/1/bin - Standard Differential Evolution Algorithm
% PARAMETER OPTIMIZATION: POP_SIZE=100, F=0.5, CR=1.0 for aggressive convergence on Rastrigin.
% --- 1. Problem Parameters Definition ---
N_DIM = 10;    
LB = -5.12;    
UB = 5.12;     
% --- 2. DE Algorithm Parameters Definition ---
POP_SIZE = 100; % Tăng kích thước quần thể
MAX_GEN = 1000;
CR = 0.5;       % Tối đa hóa tỷ lệ lai ghép
F = 0.5;        % Giảm F để tăng khai thác
% --- 3. Population Initialization (Generation 0) ---
Pop = LB + rand(POP_SIZE, N_DIM) .* (UB - LB);
Fitness = zeros(POP_SIZE, 1);
fitness_history = zeros(MAX_GEN + 1, 1);
for i = 1:POP_SIZE
   Fitness(i) = rastrigin_function(Pop(i, :));
end
[best_fitness, best_idx] = min(Fitness);
best_solution = Pop(best_idx, :);
fitness_history(1) = best_fitness;
fprintf('\nStarting Optimization: Standard DE (rand/1) | MAX_GEN=%d, POP_SIZE=%d, F=%.1f, CR=%.1f\n', MAX_GEN, POP_SIZE, F, CR);
disp('=====================================');
disp('| Generation | Best Fitness |');
disp('=====================================');
fprintf('| %-10d | %-12.8f |\n', 0, best_fitness);
% --- 4. Main Evolutionary Loop ---
for gen = 1:MAX_GEN
   NewPop = zeros(POP_SIZE, N_DIM);
   NewFitness = zeros(POP_SIZE, 1);
  
   for i = 1:POP_SIZE
       Target = Pop(i, :);
      
       % Mutation (DE/rand/1)
       indices = 1:POP_SIZE; indices(i) = [];
       r = indices(randperm(POP_SIZE - 1, 3));
       r1 = r(1); r2 = r(2); r3 = r(3);
      
       V = Pop(r3, :) + F * (Pop(r2, :) - Pop(r1, :));
       V(V < LB) = LB; V(V > UB) = UB;
      
       % Binomial Crossover (bin)
       Trial = Target;
       j_rand = randi(N_DIM);
      
       for j = 1:N_DIM
           if rand() < CR || j == j_rand
               Trial(j) = V(j);
           end
       end
      
       % Selection (Trial vs Target)
       TrialFitness = rastrigin_function(Trial);
      
       if TrialFitness < Fitness(i)
           NewPop(i, :) = Trial;
           NewFitness(i) = TrialFitness;
       else
           NewPop(i, :) = Pop(i, :);
           NewFitness(i) = Fitness(i);
       end
   end
  
   Pop = NewPop;
   Fitness = NewFitness;
  
   % Update global best solution
   [current_best_f, current_best_idx] = min(Fitness);
   if current_best_f < best_fitness
       best_fitness = current_best_f;
       best_solution = Pop(current_best_idx, :);
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

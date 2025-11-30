% --- compare_de_variants_plot_only.m ---
% Script to compare the performance and plot the convergence of 3 DE variants:
% Standard DE (Optimized params), jDE, and JADE on the Rastrigin function.
clear; close all; clc;
% --- 1. Set Statistical and Algorithm Parameters ---
NUM_RUNS = 30;           % Number of independent runs (statistical standard)
MAX_GEN = 1000;          % Maximum generations
% --- 2. List of Comparison Algorithms ---
de_functions = {
   @de_rastrigin_rand1,        'Standard DE (rand/1) Optimized';
   @de_rastrigin_jde,          'jDE (Self-Adaptive)';
   @de_rastrigin_jade,         'JADE (Adaptive pbest)';
};
NUM_VARIANTS = size(de_functions, 1);
all_history = zeros(MAX_GEN + 1, NUM_VARIANTS, NUM_RUNS);
% --- 3. Main Statistical Loop (Data Collection) ---
disp('STARTING DATA COLLECTION (30 runs per algorithm)...');
for k = 1:NUM_VARIANTS
   func_handle = de_functions{k, 1};
   func_name = de_functions{k, 2};
   fprintf('\n>>> Collecting data for algorithm: %s\n', func_name);
  
   for run = 1:NUM_RUNS
       % Giả định 3 file DE đã được cập nhật logic và tham số mới nhất
       [~, ~, history] = func_handle();
      
       if length(history) == MAX_GEN + 1
            all_history(:, k, run) = history;
       end
   end
   fprintf('  Finished data collection for %s.\n', func_name);
end
disp('\n--- DATA COLLECTION COMPLETE ---');
% --- 4. Plot Convergence Graph (Đã điều chỉnh) ---
figure;
hold on;
% Tính toán lịch sử hội tụ trung bình trên tất cả các lần chạy
avg_history = mean(all_history, 3);
generations = 0:MAX_GEN;
colors = lines(NUM_VARIANTS);
for k = 1:NUM_VARIANTS
   plot(generations, avg_history(:, k), 'Color', colors(k,:), 'LineWidth', 2, 'DisplayName', de_functions{k, 2});
end
title('Convergence Comparison on Rastrigin (Linear Scale Y, Tick Spacing 10)');
xlabel('Generation');
ylabel('Average Best Fitness so far');
% *** THAY ĐỔI CỐ ĐỊNH TRỤC Y VÀ TRỤC X ***
% 1. Trục Y: Phạm vi 0-100, khoảng cách 10
ylim([0, 100]);
yticks(0:10:100);
% 2. Trục X: Phạm vi 0-1000, khoảng cách 100
xlim([0, MAX_GEN]);
xticks(0:100:MAX_GEN);
legend('show', 'Location', 'southwest');
grid on;
hold off;

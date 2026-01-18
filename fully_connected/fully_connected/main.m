% main.m
clc; clear all; close all;
addpath("functions/");
para = para_init();

% 1. Khởi tạo kịch bản (K người dùng)
user_r = rand(para.K, 1) * 10 + 5; 
user_theta = sort(rand(para.K, 1) * pi); 
[H] = generate_channel(para, user_r, user_theta);

% 2. Chạy hội tụ TRƯỚC khi dùng DE (Sử dụng hàm gốc)
fprintf('Đang chạy thuật toán gốc...\n');
[R_old, ~, ~, ~] = algorithm_FDA_penalty_new(para, H, user_r, user_theta);

% 3. Chạy hội tụ SAU khi dùng DE (Sử dụng hàm có DE khởi tạo)
fprintf('Đang chạy thuật toán với DE...\n');
[R_new, ~, ~, ~] = algorithm_FDA_penalty_DE(para, H, user_r, user_theta);

% 4. Vẽ biểu đồ so sánh
figure;
plot(R_old, '-bo', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Khởi tạo PNF (Gốc)');
hold on;
plot(R_new, '-rs', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Khởi tạo DE (Cải tiến)');

xlabel('Số vòng lặp (Outer-loop iterations)', 'Interpreter', 'Latex');
ylabel('Tốc độ tổng (Sum Rate - bps/Hz)', 'Interpreter', 'Latex');
title('So sánh tốc độ hội tụ: Trước và Sau khi dùng DE', 'Interpreter', 'Latex');
grid on;
legend('Location', 'southeast');

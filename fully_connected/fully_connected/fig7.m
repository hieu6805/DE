clc
clear all
close all
addpath("functions/");
para = para_init();
theta = 45*pi/180; % user direction
r = 10; % user distance
para.N_T = 16; % number of TTDs
para.M = 256; % number of subcarriers

figure('Position', [100, 50, 800, 900]); % Tăng kích thước cửa sổ

%% Bandwidth B = 10 GHz
B = 1e10; 
m = 1:para.M;
para.fm_all =  para.fc + B*(2*m-1-para.M) / (2*para.M); 

% Cập nhật hàm gọi beampattern để lấy thêm P_JADE
[P_prop, P_prop_robust, P_conv_CF, P_con_MCCM, P_conv_MCM, P_DE, P_jDE, P_JADE] = beampattern(para, theta, r);

subplot(3,1,1); hold on; box on;

% 1. Vẽ đường JADE (Màu Xanh lá - Nét đậm nhất - Tốt nhất)
plot(para.fm_all/1e9, 10*log10(P_JADE/para.N), '-', 'Color', [0.4660 0.6740 0.1880], 'LineWidth', 2.0);

% 2. Vẽ đường jDE (Màu Tím)
plot(para.fm_all/1e9, 10*log10(P_jDE/para.N), '--', 'Color', [0.4940 0.1840 0.5560], 'LineWidth', 1.5);

% 3. Vẽ đường DE (Màu Cam)
plot(para.fm_all/1e9, 10*log10(P_DE/para.N), ':', 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 1.5);

% 4. Vẽ đường PNF gốc (Xanh dương - Nét đứt)
plot(para.fm_all/1e9, 10*log10(P_prop/para.N), '-.', 'Color', [0 0.4470 0.7410], 'LineWidth', 1.5);

% Các đường tham chiếu khác (Vẽ mờ hơn)
plot(para.fm_all/1e9, 10*log10(P_prop_robust/para.N), ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.2);
plot(para.fm_all/1e9, 10*log10(P_con_MCCM/para.N), '--', 'Color', 'k', 'LineWidth', 0.8);
plot(para.fm_all/1e9, 10*log10(P_conv_MCM/para.N), '--', 'Color', 'm', 'LineWidth', 0.8);
plot(para.fm_all/1e9, 10*log10(P_conv_CF/para.N), ':', 'Color', 'c', 'LineWidth', 0.8);

set(gca, 'TickLabelInterpreter', 'latex');
% Cập nhật Legend
legend("Proposed JADE (Best)", "Proposed jDE", "Proposed DE", "TTD-BF, PNF", ...
       "TTD-BF, Robust", "Conv, MCCM", "Conv, MCM", "Conv, CF", ...
       'Interpreter', 'Latex', 'Location', 'southwest', 'NumColumns', 2);

xlabel('Frequency (GHz)', 'Interpreter', 'Latex');
ylabel('Norm. Array Gain (dB)', 'Interpreter', 'Latex');
title(['$B = ' num2str(B/1e9) '$ GHz'], 'Interpreter', 'Latex');
ylim([-5 0.5]); 

%% Bandwidth B = 20 GHz
B = 2e10; 
m = 1:para.M;
para.fm_all =  para.fc + B*(2*m-1-para.M) / (2*para.M); 

[P_prop, P_prop_robust, P_conv_CF, P_con_MCCM, P_conv_MCM, P_DE, P_jDE, P_JADE] = beampattern(para, theta, r);

subplot(3,1,2); hold on; box on;
plot(para.fm_all/1e9, 10*log10(P_JADE/para.N), '-', 'Color', [0.4660 0.6740 0.1880], 'LineWidth', 2.0);
plot(para.fm_all/1e9, 10*log10(P_jDE/para.N), '--', 'Color', [0.4940 0.1840 0.5560], 'LineWidth', 1.5);
plot(para.fm_all/1e9, 10*log10(P_DE/para.N), ':', 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 1.5);
plot(para.fm_all/1e9, 10*log10(P_prop/para.N), '-.', 'Color', [0 0.4470 0.7410], 'LineWidth', 1.5);
plot(para.fm_all/1e9, 10*log10(P_prop_robust/para.N), ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.2);
plot(para.fm_all/1e9, 10*log10(P_con_MCCM/para.N), '--', 'Color', 'k', 'LineWidth', 0.8);
plot(para.fm_all/1e9, 10*log10(P_conv_MCM/para.N), '--', 'Color', 'm', 'LineWidth', 0.8);
plot(para.fm_all/1e9, 10*log10(P_conv_CF/para.N), ':', 'Color', 'c', 'LineWidth', 0.8);

set(gca, 'TickLabelInterpreter', 'latex');
xlabel('Frequency (GHz)', 'Interpreter', 'Latex');
ylabel('Norm. Array Gain (dB)', 'Interpreter', 'Latex');
title(['$B = ' num2str(B/1e9) '$ GHz'], 'Interpreter', 'Latex');
ylim([-10 1]); 

%% Bandwidth B = 30 GHz
B = 3e10; 
m = 1:para.M;
para.fm_all =  para.fc + B*(2*m-1-para.M) / (2*para.M); 

[P_prop, P_prop_robust, P_conv_CF, P_con_MCCM, P_conv_MCM, P_DE, P_jDE, P_JADE] = beampattern(para, theta, r);

subplot(3,1,3); hold on; box on;
plot(para.fm_all/1e9, 10*log10(P_JADE/para.N), '-', 'Color', [0.4660 0.6740 0.1880], 'LineWidth', 2.0);
plot(para.fm_all/1e9, 10*log10(P_jDE/para.N), '--', 'Color', [0.4940 0.1840 0.5560], 'LineWidth', 1.5);
plot(para.fm_all/1e9, 10*log10(P_DE/para.N), ':', 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 1.5);
plot(para.fm_all/1e9, 10*log10(P_prop/para.N), '-.', 'Color', [0 0.4470 0.7410], 'LineWidth', 1.5);
plot(para.fm_all/1e9, 10*log10(P_prop_robust/para.N), ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.2);
plot(para.fm_all/1e9, 10*log10(P_con_MCCM/para.N), '--', 'Color', 'k', 'LineWidth', 0.8);
plot(para.fm_all/1e9, 10*log10(P_conv_MCM/para.N), '--', 'Color', 'm', 'LineWidth', 0.8);
plot(para.fm_all/1e9, 10*log10(P_conv_CF/para.N), ':', 'Color', 'c', 'LineWidth', 0.8);

set(gca, 'TickLabelInterpreter', 'latex');
xlabel('Frequency (GHz)', 'Interpreter', 'Latex'); 
ylabel('Norm. Array Gain (dB)', 'Interpreter', 'Latex');
title(['$B = ' num2str(B/1e9) '$ GHz'], 'Interpreter', 'Latex');
ylim([-15 1]);
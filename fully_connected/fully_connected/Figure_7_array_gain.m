% Figure_7_array_gain_DE.m
clc; clear all; close all;
addpath("functions/");

para = para_init();

% --- ĐỒNG BỘ HÓA THAM SỐ CHO 1 NGƯỜI DÙNG (IMPORTANT) ---
para.K = 1;         % Số lượng người dùng = 1
para.N_RF = 1;      % Số lượng RF chain = 1
theta_val = 45*pi/180; 
r_val = 10;            
user_theta = theta_val; 
user_r = r_val;
% -------------------------------------------------------

para.N_T = 16;     
para.M = 256;      
Bandwidths = [10e9, 20e9, 30e9];
figure;

for i = 1:3
    para.B = Bandwidths(i);
    para.fm_all = para.fc + linspace(-para.B/2, para.B/2, para.M);
    
    % Tạo kênh H với kích thước (N, K, M) tương ứng (N, 1, 256)
    H = zeros(para.N, para.K, para.M);
    for m = 1:para.M
        H(:,1,m) = array_response_vector(user_r, user_theta, para.N, para.d, para.fm_all(m));
    end
    
    % 1. Chạy thuật toán PNF gốc
    % Hàm này bên trong sẽ gọi RWMMSE, giờ para.K=1 nên sẽ không lỗi index
    [~, A_PNF, ~, t_PNF] = algorithm_HTS_PNF(para, H, user_r, user_theta);
    
    % 2. Chạy thuật toán DE bạn đã bổ sung
    [~, A_DE, ~, t_DE] = algorithm_HTS_PNF_DE(para, H, user_r, user_theta);
    
    % Tính toán Array Gain
    P_prop = zeros(para.M, 1);
    P_DE = zeros(para.M, 1);
    
    for m = 1:para.M
        fm = para.fm_all(m);
        bm = array_response_vector(r_val, theta_val, para.N, para.d, fm);
        
        % Lấy analog beamformer (cột 1)
        am_pnf = A_PNF(:, 1, m); 
        P_prop(m) = abs(bm' * am_pnf)^2; 
        
        am_de = A_DE(:, 1, m);
        P_DE(m) = abs(bm' * am_de)^2;
    end

    % Vẽ đồ thị
    subplot(3,1,i); hold on; box on;
    plot(para.fm_all/1e9, 10*log10(P_prop/para.N), '--k', 'LineWidth', 1.2, 'DisplayName', 'Original PNF');
    plot(para.fm_all/1e9, 10*log10(P_DE/para.N), '-r', 'LineWidth', 1.5, 'DisplayName', 'Proposed DE');
    
    ylabel('Normalized Gain (dB)', 'Interpreter', 'Latex');
    title(['$B = ', num2str(para.B/1e9), '$ GHz'], 'Interpreter', 'Latex');
    if i == 3, xlabel('Frequency (GHz)', 'Interpreter', 'Latex'); end
    legend('Location', 'southwest');
    grid on;
end
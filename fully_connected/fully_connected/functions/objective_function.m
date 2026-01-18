function obj = objective_function(t_vector, tau_prop, fm_all)
    % Hàm mục tiêu: Tối đa hóa tổng độ lớn phản hồi mảng trên mọi tần số
    % Sum over M: | Sum over N_T: exp( -j*2*pi*f * (tau_prop + t_opt) ) |
    
    % Tính phase term: 2 * pi * f_m * (tau_prop_l + t_l)
    % Ma trận hóa: (tau_prop + t_vector) là Nx1, fm_all là 1xM
    time_total = tau_prop + t_vector; % N_T x 1
    
    % Tính toán vector hóa để nhanh hơn
    % Phase matrix: N_T x M
    phase_mat = -1i * 2 * pi * (time_total * fm_all); 
    
    % Sum over antennas (dim 1) -> result is 1 x M complex values
    array_response = sum(exp(phase_mat), 1);
    
    % Sum of magnitudes (Objective to maximize)
    obj = sum(abs(array_response));
end

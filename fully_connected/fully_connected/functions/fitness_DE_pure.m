function fitness = fitness_DE_pure(t_vector, para, H, A_PS)
    N = para.N; N_T = para.N_T; N_RF = para.N_RF; K = para.K;
    t = reshape(t_vector, [N_T, N_RF]);
    e_sub = ones(N/N_T, 1);
    
    W_hybrid = zeros(N, K, para.M);
    for m = 1:para.M
        fm = para.fm_all(m);
        % Kết hợp Phase Shifter cố định và TTD đang tối ưu
        Am = A_PS .* kron(exp(-1j * 2 * pi * fm * t), e_sub);
        
        % Tính Digital Beamformer (Zero-Forcing)
        Heq = Am' * H(:,:,m);
        % Regularized Zero-Forcing
        Dm = Heq' / (Heq * Heq' + (K / para.Pt) * eye(K));
        
        % Chuẩn hóa công suất (Power Constraint)
        Wm = Am * Dm;
        Wm = Wm / norm(Wm, 'fro') * sqrt(para.Pt / para.M);
        W_hybrid(:,:,m) = Wm;
    end
    
    % Tính SE tổng cộng (Sử dụng hàm gốc của bạn để đảm bảo so sánh đúng)
    R_sum = rate_fully_digital(para, W_hybrid, H); 
    fitness = -R_sum / (para.M + para.Lcp); 
end
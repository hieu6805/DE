function fitness = fitness_DE_TTD(t_vector, para, H, A_PS)
    % Lấy kích thước từ tham số hệ thống
    N = para.N;
    N_T = para.N_T;
    N_RF = para.N_RF;
    K = para.K;
    
    % Chuyển vector cá thể về dạng ma trận TTD: N_T x N_RF
    t = reshape(t_vector, [N_T, N_RF]);
    
    R_sum = 0;
    e_sub = ones(N/N_T, 1); % Vector hỗ trợ kron để mở rộng TTD cho các anten
    
    for m = 1:para.M
        fm = para.fm_all(m);
        
        % Tính ma trận Analog Beamformer Am (kết hợp PS và TTD)
        % Dựa trên logic hàm analog_bamformer trong file của bạn [cite: 145]
        Tm = exp(-1j * 2 * pi * fm * t);
        Am = A_PS .* kron(Tm, e_sub); 
        
        % Tính Digital Beamformer (Zero-Forcing)
        % Sử dụng 1 làm noise power và Pt để điều chỉnh theo update_P [cite: 112]
        Heq = Am' * H(:,:,m); 
        Dm = Heq' / (Heq * Heq' + (K / para.Pt) * eye(K));
        
        % Chuẩn hóa công suất cho subcarrier m
        Wm = Am * Dm;
        Wm = Wm / norm(Wm, 'fro') * sqrt(para.Pt / para.M);
        
        % Tính Rate cho subcarrier m (Noise power = 1 theo rate_single )
        for k = 1:K
            h_km = H(:,k,m);
            w_km = Wm(:,k);
            % Nhiễu + Noise (với noise power mặc định là 1)
            interference = norm(h_km' * Wm(:, [1:k-1, k+1:end]))^2 + 1;
            R_sum = R_sum + log2(1 + abs(h_km' * w_km)^2 / interference);
        end
    end
    
    % Giá trị fitness (tối đa hóa SE nên trả về giá trị âm)
    fitness = -R_sum / (para.M + para.Lcp); 
end
%% RWMMSE method for optimizing the digital beamformer
function [R, D] = RWMMSE(para, H, H_equal, D, A)
    R_pre = 0;
    for i = 1:20
        E = eye(para.K);
        Phi = 0; Upsilon = 0;
        for k = 1:para.K
            hk = H_equal(:,k);
            dk = D(:,k); 
            I = norm(hk'*D)^2 + norm(A*D, 'fro')^2/para.Pt; 
            w_k = 1 + abs(hk'*dk)^2 / (I - abs(hk'*dk)^2);
            v_k = hk'*dk / I;
        
            Phi = Phi + w_k*abs(v_k)^2 * ( hk*hk' + eye(para.N_RF)/para.Pt );
            Upsilon = Upsilon + w_k*conj(v_k)*E(:,k)*hk';
        end
        
        D = Phi\Upsilon';
        % check convergence
        [R] = rate_single_carrier(para, A*D, H);
        if abs(R - R_pre)/R <= 1e-4
            break;
        end
        R_pre = R;
    end
end

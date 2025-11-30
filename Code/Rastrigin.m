function f = Rastrigin(x)
% f = A*n + sum(x.^2 - A*cos(2*pi*x))
A = 10;
if ismatrix(x) && size(x,1) > 1
    % matrix: each row is a solution
    n = size(x,2);
    f = A*n + sum(x.^2 - A*cos(2*pi*x),2);
else
    % vector
    n = numel(x);
    f = A*n + sum(x.^2 - A*cos(2*pi*x));
end
end



function y = rastrigin_function(x)
% RASTRIGIN_FUNCTION: Calculates the value of the n-dimensional Rastrigin function.
% The objective is to find the global minimum at y = 0 when x = [0, 0, ..., 0].
%
% Formula: f(x) = A*n + sum_{i=1}^{n} [x_i^2 - A*cos(2*pi*x_i)]

A = 10;
n = length(x);

sum_part = sum(x.^2 - A * cos(2 * pi * x));

y = A * n + sum_part;

end

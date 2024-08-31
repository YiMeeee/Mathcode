function dx = lorenz_system(t, x, sigma, rho, beta)
    dx = zeros(3,1); % 初始化输出为列向量
    dx(1) = sigma * (x(2) - x(1));
    dx(2) = x(1) * (rho - x(3)) - x(2);
    dx(3) = x(1) * x(2) - beta * x(3);
end

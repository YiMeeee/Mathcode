% 洛伦兹系统参数
sigma = 10;
rho = 28;
beta = 8/3;

% 初始条件和时间跨度
x0 = [1; 1; 1];
tspan = [0 10];

% 使用 ode45 求解方程
[t, x] = ode45(@(t, x) lorenz_system(t, x, sigma, rho, beta), tspan, x0);

% 绘制结果
figure;
plot3(x(:,1), x(:,2), x(:,3));
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Lorenz Attractor');
grid on;

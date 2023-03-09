clear all
clc

% Define the input-output pairs
x = [0; 0.8; 1.6; 3; 4; 5];
y = [0.5; 1; 4; 5; 6; 9];

% Add a column of ones to x to represent the bias term
X = [ones(length(x),1), x];

% Solve for w using the LLS method
w = inv(X' * X) * X' * y;

% Extract the intercept term (b)
b = w(1);

% Extract the slope (w)
k = w(2);

% Print the results
fprintf('Intercept (b): %f\n', b);
fprintf('Slope (w): %f\n', k);

% Plotting the fitting results
figure;
for i=1:length(x)
    plot(x(i),y(i),"o",'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'k');
    hold on;
end
q = x(1):0.05:x(length(x));
p = k*q+b;
plot(q,p,'-r', 'LineWidth', 2);
hold off;


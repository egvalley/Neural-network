clear all
clc

% Define the input-output pairs
x = [0; 0.8; 1.6; 3; 4; 5];
y = [0.5; 1; 4; 5; 6; 9];


% Initialize weights and biases
w = rand();
b = rand();

% Set learning rate and number of epochs
lr = 0.001;
epochs = 10000;

% Initialize arrays to store weights and biases
w_arr = zeros(1, epochs);
b_arr = zeros(1, epochs);

% Loop for number of epochs
for i = 1:epochs
    % Loop for each data point
    for j = 1:length(x)
        
        % Calculate predicted output
        y_pred = w * x(j) + b;
        
        % Update weights and biases
        w = w + lr * (y(j) - y_pred) * x(j);
        b = b + lr * (y(j) - y_pred);
        
        % Store updated weights and biases
        w_arr(i) = w;
        b_arr(i) = b;
    end
end


% Plot the fitting result
figure;
subplot(3,1,1);
plot(x, y, 'o', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'k');
hold on;
x_fit = 0:0.1:5;
y_fit = w * x_fit + b;
plot(x_fit, y_fit, '-r', 'LineWidth', 2);
xlabel('Input (x)');
ylabel('Output (y)');
legend('Data', 'Fitted Line');

subplot(3,1,2);
plot(1:epochs, w_arr, '-b', 'LineWidth', 2);
xlabel('Epoch');
ylabel('Weight (w)');
subplot(3,1,3);
plot(1:epochs, b_arr, '-g', 'LineWidth', 2);
xlabel('Epoch');
ylabel('Bias (b)');

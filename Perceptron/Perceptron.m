clear all
clc

% Input data
X = [0 0;0 1;1 0;1 1];
y = [0; 1; 1; 1];

% Change the label from 0 to -1
for i = 1:4
    if y(i)== 0
        y(i)=-1;
    end
end

% Initial weights
w = rand(2, 1);
b = rand();

% Create predicting matrix for storage
y_pred = rand(4,1);

% Learning rate
eta =1;

% Recording the iteration number
Iter_num=0;

% Perceptron loop
while true
    for i = 1:4
        y_pred(i) = sign(X(i, :) * w + b);
        if (y_pred(i) ~= y(i))
            w = w + eta * (y(i) - y_pred(i)) * X(i, :)'; % Updating the weights
            b = b + eta * (y(i) - y_pred(i));
        end
    end

    % Conditions for stopping iterations
    if isequal(y_pred,y)
        break
    elseif Iter_num > 1000
        break
    end

    % Trajectories of weights    
    Iter_num=Iter_num+1;    
    w_trajectory(Iter_num, :) = w';
    b_trajectory(Iter_num, :) = b;
end

% Plot the trajectories of weights
figure;
subplot(2, 2, 1);
plot(w_trajectory(:, 1), '-o');
xlabel('Iteration');
ylabel('w1');
title('Trajectories of weights');

subplot(2, 2, 2);
plot(w_trajectory(:, 2), '-o');
xlabel('Iteration');
ylabel('w2');
title('Trajectories of weights');

subplot(2, 2, 3);
plot(b_trajectory, '-o');
xlabel('Iteration');
ylabel('b');
title('Trajectories of bias');

% Plotting the illustrating figure
subplot(2,2,4);
plot(0,0,"*",1,0,"o",0,1,"o",1,1,"*",'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'k');
hold on;
p = 0:0.05:1;
q = -(w(1,1)*p+b)/w(2,1);
plot(p,q,'-r', 'LineWidth', 2);



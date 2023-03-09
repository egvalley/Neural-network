clear all
clc

% Define the Rosenbrock function
rosenbrock = @(x,y) 100 * (y - x^2)^2 + (1 - x)^2;

% Define the gradient of the Rosenbrock function
rosenbrock_grad = @(x,y) [-400*x*(y-x^2)-2*(1-x); 200*(y-x^2)];

% Set the initial point
x=0;
y=0.5;

% Set the tolerance level
tol = 1e-5;

% Set the step size (also known as the learning rate)
alpha = 0.001;

% Initialize the iteration counter
Num_k = 1;

% Initialize the trajectory matrix
x_trajectory=[];
y_trajectory=[];
z_trajectory=[];

% Start the steepest descent algorithm
while rosenbrock(x,y) > tol

    % Record the trajectory 
    x_trajectory(Num_k,1) = x;
    y_trajectory(Num_k,1) = y;
    z_trajectory(Num_k,1) = rosenbrock(x,y); 

    % Compute the gradient
    p = rosenbrock_grad(x,y);
    
    % Update the current point
    x = x - alpha*p(1);
    y = y - alpha*p(2);

    % Increment the iteration counter
    Num_k = Num_k + 1;

end
    % Record the last trajectory 
    x_trajectory(Num_k,1) = x;
    y_trajectory(Num_k,1) = y;
    z_trajectory(Num_k,1) = rosenbrock(x,y);


% Plot the trajectories 
figure;
plot3(x_trajectory,y_trajectory,z_trajectory,'o', 'MarkerSize', 6, 'MarkerFaceColor', 'red','MarkerEdgeColor', 'red');
grid on;
hold on;

% Plot the valley function
[m,n] = meshgrid(-1:0.1:1.5);
q = (1-m).^2 + 100*(n-m.^2).^2;
surf(m,n,q);
view(110,50);

% Set the figure
xlabel('x');
ylabel('y');
zlabel('z');
title('Trajectories of x y z');

% Plot the x y in 2D
figure;
plot(x_trajectory,y_trajectory, '-o');
xlabel('x');
ylabel('y');
title('Trajectories of x y');

% Display the minimum value and the number of iterations
fprintf('Minimum value: %f\n', rosenbrock(x,y));
fprintf('Number of iterations: %d\n', Num_k);

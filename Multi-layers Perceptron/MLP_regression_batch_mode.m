clear all
clc

load('traindata.mat');
load('testdata.mat');
load('testdata2.mat');

% Set the train dataset
x=traindata(:,1)';
t=traindata(:,2)';

% Specify the structure and learning algorithm for MLP
net = fitnet(20,'trainlm'); 
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'purelin';
net = configure(net,x,t);
net.trainparam.lr=0.01;
net.trainparam.epochs=10000;
net.trainparam.goal=1e-8;
net.divideParam.trainRatio=1.0;% all data for training
net.divideParam.valRatio=0.0;
net.divideParam.testRatio=0.0;

% Train the network
net = train(net, x, t);
view(net)

% Generate the test input data
input=testdata2(:,1)';
desiredout=testdata2(:,2)';

% Feed the input
pred = net(input);
perf = perform(net, desiredout, pred);

% Plot the fitting results and compare
figure;
plot(input',pred','*','MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b','MarkerSize',10);% Plot the fitting dataset
hold on;
plot(testdata2(:,1),testdata2(:,2),'o','MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r','MarkerSize',5);% Plot the original test dataset

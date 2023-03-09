clear all
clc

% Read directory
files1=dir('C:\Users\sunsh\Desktop\tem\tem matlab photo\group_1\train');
folder1 = 'C:\Users\sunsh\Desktop\tem\tem matlab photo\group_1\train';

% Define the magnifying/shrinking coefficent
coeff = 0.25;

% Read directory
files2=dir('C:\Users\sunsh\Desktop\tem\tem matlab photo\group_1\test');
folder2 = 'C:\Users\sunsh\Desktop\tem\tem matlab photo\group_1\test';

% Read images and processing
[train_images,trainDesired]=loadimage(files1,folder1,coeff);
[test_images,testDesired]=loadimage(files2,folder2,coeff);

% Set the epoch number and get the training images' number
epochs = 40; 
train_num = length(files1)-2;


% 1. Change the input to cell array form for sequential training 
 images_c = num2cell(train_images, 1);
 labels_c = num2cell(trainDesired, 1);
 
 % 2. Construct and configure the MLP
 net = patternnet(8);
 net.divideFcn = 'dividetrain'; % input for training only
 net.performParam.regularization = 0.1; % regularization strength
 net.trainFcn = 'traingdx'; % 'trainrp' 'traingdx'
 net.trainParam.epochs = epochs;
 

 % 3. Train the network in sequential mode
 for i = 1 : epochs
 
     display(['Epoch: ', num2str(i)])
     
     idx = randperm(train_num); % shuffle the input
     
     net = adapt(net, images_c(:,idx), labels_c(:,idx));
 
 end

% Feed the testing or the training images
test_pred_output = round(net(test_images));
train_pred_ouput = round(net(train_images));

disp("============test");
calacc_classification(testDesired,test_pred_output,length(files2)-2);
disp("============train");
calacc_classification(trainDesired,train_pred_ouput,length(files1)-2);
disp("============");

function calacc_classification(Desired,pred_output,length)

% Record the successful prediction number
k1=0;
k2=0;

% Identify the successful predictions of input
for i = 1:length
    if pred_output(i)==0
        if Desired(i) == pred_output(i)
            k1=k1+1;
        end
    end
    if pred_output(i)==1
        if Desired(i) == pred_output(i)
            k2=k2+1;
        end
    end
end

% Calculate the accuracy and Display the accuracy
accuracy = (k1+k2)/length;
Z=['Accuracy ',num2str(accuracy)];
X=['there are ',num2str(k1), ' successful predictions of man-made scenes in ',num2str(length),' input cases'];
Y=['there are ',num2str(k2), ' successful predictions of natural secenes in ',num2str(length),' input cases'];
disp(Z);
disp(X);
disp(Y);
end

function [I_reconstructed] = svd_process(I)

    % Perform SVD on the image
    [U, S, V] = svd(I);

    % Calculate the total energy of the singular values
    energy = sum(diag(S).^2);

    % Calculate the cumulative energy of the singular values
    cumulative_energy = cumsum(diag(S).^2);

    % Calculate the percentage of the information to keep
    threshold = 0.9;
    n = find(cumulative_energy >= threshold*energy, 1, 'first');

    % Use only the first n singular values to reconstruct the image
    I_reconstructed = U(:,1:n) * S(1:n,1:n) * V(:,1:n)';

end

function [image,label]=loadimage(files,folder,coeff)
    for i=3:length(files)
        % Get the label
        tmp1 = strsplit(files(i).name, {'_', '.'});
        label(1,i-2)= str2num(tmp1{2});
        % Read train images and store
        filename = fullfile(folder, files(i).name);
        % Change the format 
        images = double(imread(filename));
        % Compress the pictures
        resizeImage = imresize(images,coeff);
        % SVD compressing
        svdimage=svd_process(resizeImage);
        % Put the data into columns
        V = svdimage(:);
        image(:,i-2)=V;      
    end
end

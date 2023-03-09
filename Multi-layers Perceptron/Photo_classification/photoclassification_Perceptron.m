clear all
clc

% Read directory
files1=dir('C:\Users\sunsh\Desktop\tem\tem matlab photo\group_1\train');
folder1 = 'C:\Users\sunsh\Desktop\tem\tem matlab photo\group_1\train';

% Define the magnifying/shrinking coefficent
coeff = 0.125;

% Read directory
files2=dir('C:\Users\sunsh\Desktop\tem\tem matlab photo\group_1\test');
folder2 = 'C:\Users\sunsh\Desktop\tem\tem matlab photo\group_1\test';

% Read images and processing
[train_images,trainDesired]=loadimage(files1,folder1,coeff);
[test_images,testDesired]=loadimage(files2,folder2,coeff);

% Initial weights
w = rand(size(train_images,2),1);
b = rand();
train_pred = rand(1,length(files1)-2);

% Learning rate
eta =5;

% Counting number
numi=0;

while true
    % Update the weights
    for i = 1:length(files1)-2
        train_pred(i) = sign(train_images(i,:)*w + b);
        if (train_pred(i) ~= trainDesired(i))
            w = w + eta * (trainDesired(i) - train_pred(i)) * train_images(i,:)';
            b = b + eta * (trainDesired(i) - train_pred(i));
        end
    end

    % make the judgement
    if isequal(train_pred,trainDesired)
        break;
    elseif numi > 70
        break;
    end

    % Trajectories 
    numi=numi+1; 
    disp(numi);

end

disp("============");


% Identify the successful predictions of input
for k = 1:length(files2)-2
    test_pred_output(k) = sign(test_images(k,:)*w + b);
end

% Displaying the results and evaluations
disp("============test");
calacc_classification(testDesired,test_pred_output,length(files2)-2);
disp("============train");
calacc_classification(trainDesired,train_pred,length(files1)-2);
disp("============");

% Evaluation function
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

% Local field function
function [output]=sign(x)

    if x<0
        output=0;
    elseif x>0
        output=1;
    else
        output=0.5;
    end
end

% SVD proceeding
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

% Loading the image and getting the labels
function [image,label]=loadimage(files,folder,coeff)
    for i=3:length(files)
        % Get the label
        tmp1 = strsplit(files(i).name, {'_', '.'});
        label(i-2)= str2num(tmp1{2});
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
        image(i-2,:)=V;      
    end
end

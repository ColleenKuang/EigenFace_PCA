clear 
clc
%Import Image Data%
imgPath = './Portrait/';      
imgDir  = dir([imgPath '*.jpg']);
%Resize Image to 360*440
for i = 1:length(imgDir)          
    imgpath = [imgPath imgDir(i).name];
    img = imread(imgpath); %read image
    img = im2double(rgb2gray(img)); % convert image to gray scale and then to double precision
    img = imresize(img,[360 440]); %resize the image to 360*440
    [r,c] = size(img); % get number of rows and columns in image
    I(:,i) = img(:); % convert image to vector and store as column in matrix I
end
% calculate mean image
I_mean = mean(I,2);
% subtract mean image from the set of images
I_shifted = I - repmat(I_mean,1,length(imgDir));
%perform PCA. 
%Matrix I was used as input instead of I_shifted 
%because Matlab documentation states that pca function centers the data
[coeff,score,latent,~,explained,mu] = pca(I);
%calculate eigenfaces
eigFaces = I_shifted * coeff;
% put eigenface in array and display
figure;
ef = [];
for row = 1:6
    eftemp = [];
    for n = 1:5
      temp = reshape(eigFaces(:,(row - 1) * 5 + n),r,c);
      temp = histeq(temp,255);
      ef = [ef temp];
      eftemp = [eftemp temp];
    end
    subplot(6,1,row);
    imshow(eftemp,'Initialmagnification','fit');
    if row == 1
        title('EigenFace');
    end
end
%Reconstruction
M = length(imgDir) ;
%calculate covariance 
cov_I = cov(I);
%Set the first image as the test image
img = imread('./Portrait/1.jpg');
% convert to gray and then to double
img = im2double(rgb2gray(img)); 
img = imresize(img,[360 440]);
I_test = img(:); % convert image to vector
I_test = I_test - I_mean; % subtract mean images
%K = 2 reconstruct test image
[V,D] = eigs(cov_I, 2);
DD = diag(D);
v = zeros(30,1);
for i = 1:length(DD)
    v(i) = DD(i);
end
I_recon_2 = I_mean + eigFaces*v;
%reshape reconstructed test image
I_recon_2 = reshape(I_recon_2, r,c);

%K = 15 reconstruct test image
[V,D] = eigs(cov_I, 15);
DD = diag(D);
v = zeros(30,1);
for i = 1:length(DD)
    v(i) = DD(i);
end
I_recon_15 = I_mean + eigFaces*v;
%reshape reconstructed test image
I_recon_15 = reshape(I_recon_15, r,c);

%K = 30 reconstruct test image
[V,D] = eigs(cov_I, 30);
DD = diag(D);
v = zeros(30,1);
for i = 1:length(DD)
    v(i) = DD(i);
end
I_recon_30 = I_mean + eigFaces*v;
%reshape reconstructed test image
I_recon_30 = reshape(I_recon_30, r,c);
%display original and reconstructed test image
figure
subplot(2,2,1);
imshow(img);
title('Original test image');
subplot(2,2,2)
imshow(I_recon_2);
title('k = 2 Reconstructed test image');
subplot(2,2,3)
imshow(I_recon_15);
title('k = 15 Reconstructed test image');
subplot(2,2,4)
imshow(I_recon_30);
title('k = 30 Reconstructed test image');




clear;
clc;
close all;

for ite = 1:120
    num = num2str(ite);
    if length(num) == 1
        num = strcat('00', num);
    end
    if length(num) == 2
        num = strcat('0', num);
    end
    
    filename = strcat(num, '.jpg');
    %read data
    RGB = imread(filename);
    
    %extract channel data
    r = RGB(:, :, 1);
    g = RGB(:, :, 2);
    b = RGB(:, :, 3);
    
    %flatten channel data
    r = r(:);
    g = g(:);
    b = b(:);

    %convert image data to double version
    RGB = double(RGB);
    %Normalize pixel intensity
    RGB(:, :, 1) = RGB(:, :, 1)/sum(r)*500*500*128;
    RGB(:, :, 2) = RGB(:, :, 2)/sum(g)*500*500*128;
    RGB(:, :, 3) = RGB(:, :, 3)/sum(b)*500*500*128;

    RGB = uint8(RGB);
    
    filename_write = ['/Normalized/', filename];
    %mkdir Normalized;
    
    if ~exist('Normalized', 'dir')
       mkdir('Normalized')
    end
    
    imwrite(RGB,[pwd, filename_write]);    
    
end


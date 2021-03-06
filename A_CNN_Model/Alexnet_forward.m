clear all; 
close all; 

%% GET AND SHOW IMAGE 
image_data = imread('input_image.jpg'); 
[H,W,C]= size(image_data); 

imshow(image_data)

image_data = reshape(image_data,[1,H,W,C]);
image_data = permute(image_data,[1,4,2,3]);

fclose('all');

%% RETRIEVE W1 DATA
disp('Retreiving W1 data')
W1_file = fopen('./params/W1_tensorflow.bin');

% Essential Parameters 
num_filters = 96; 
num_channels = 3; 
height = 11; 
width = 11; 

W1_read_single = fread(W1_file,[num_filters*num_channels*height*width,1],'single');
W1_single = zeros(height,width,num_channels,num_filters);

Hex_W1_read_single = []; 
for i=1:1:length(W1_read_single)
    Hex_W1_read_single = [Hex_W1_read_single; dec_to_hex_single(W1_read_single(i))];
end

index = 1; 

for HH = 1:1:height 
    for WW = 1:1:width
        for channels = 1:1:num_channels
            for filters = 1:1:num_filters
                W1_single(HH,WW,channels,filters) = W1_read_single(index,1); 
                index = index + 1; 
            end
        end 
    end 
end 
W1_single = permute(W1_single,[4,3,1,2]);
fclose('all');

%% RETRIEVE b1 DATA
disp('Retreiving b1 data')

b1_file = fopen('./params/W1_tensorflow.bin');

% Essential Parameters 
num_filters = 96; 

b1_read_single = fread(b1_file,[num_filters,1],'single');
b1_single = zeros(num_filters,1);

Hex_b1_read_single = []; 
for i=1:1:length(b1_read_single)
    Hex_b1_read_single = [Hex_b1_read_single; dec_to_hex_single(b1_read_single(i))];
end

b1_single = b1_read_single; 

fclose('all');

%%CONVOLUTION 1 
conv_param = {0, 4}; 
[out, cache] = conv_forward_naive(image_data, W1_single, b1_single, conv_param); 


fclose('all'); 
clear all; 
close all; 

conv1_file = fopen('.\conv1.bin');

%% Convert Conv1 Data
disp('Converting W1 data')

num_channels = 96; 
height = 55; 
width = 55; 

conv1_read = fread(conv1_file,[num_channels*height*width,1],'single');
conv1 = zeros(height,width,num_channels);

index = 1; 

for channels = 1:1:num_channels %HH = 1:1:height 
    for HH = 1:1:height %WW = 1:1:width
        for WW = 1:1:width %channels = 1:1:num_channels
            conv1(HH,WW,channels) = conv1_read(index,1); 
            index = index + 1; 
        end 
    end 
end 

conv1_array = reshape(conv1,[height*width*num_channels,1]);

fclose('all');

conv1_permute = permute(conv1,[2,1,3,4]); 
conv1_reshape = reshape(conv1_permute,[height*width*num_channels,1]);
conv1_char = num2hex(single(conv1_reshape)); 

% conv1_char = []; 
% 
% for i = 1:1:length(conv1_reshape) 
%     conv1_char = [conv1_char; dec_to_hex_single(double(conv1_reshape(i)))]; 
% end 



disp('Writing conv1 TXT File ...')


conv1_char_size=size(conv1_char);
conv1_char_size(1);


fileID = fopen('.\conv1_coe.txt','w');
for i = 1:1:conv1_char_size(1)
    data = conv1_char(i,1:4); 
    fprintf(fileID, '%s,\n',data);
end 
fclose(fileID); 

fileID = fopen('.\conv1.txt','w');
for i = 1:1:conv1_char_size(1)
    data = conv1_char(i,1:4); 
    fprintf(fileID, '%s\n',data);
end 
fclose(fileID); 



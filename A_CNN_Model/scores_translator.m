clear all; 
close all; 
fclose('all'); 


scores_file = fopen('.\affine3.bin');

%% Get Scores Data
disp('Get Scores Data')
scores_file = fopen('.\affine3.bin');
% Essential Parameters 

num_classes = 10; 


scores_read = fread(scores_file,[num_classes,1],'double');

hex_scores_read = []; 
for i=1:1:length(scores_read)
    hex_scores_read = [hex_scores_read; dec_to_hex_single(scores_read(i))];
end

%% Convert to Floating Point Half Precision Hex
disp('Convert to Floating Point Half Precision Hex'); 

scores_half_hex = []; 

for i = 1:1:length(scores_read)
        hex = dec_to_hex_single(scores_read(i));  %
        scores_half_hex = [scores_half_hex; [hex(1:4),'0000']]; %Make data Half Precision
end 


%% Write Conv Output Truth File 
disp('Write Scores File ')
fileID = fopen('.\scores_data.txt','w');
for i = 1:1:length(scores_read)
    data = scores_half_hex(i,1:4);  
    fprintf(fileID, '%s\n',data(1:4));
end 
fclose(fileID); 

fileID = fopen('.\scores_data_coe.txt','w');
for i = 1:1:length(scores_read)
    data = scores_half_hex(i,1:4); 
    fprintf(fileID, '%s,\n',data);
end 
fclose(fileID); 
clear all;
close all;
format long;
rng(65);
%% defination
fs = 64e9;
resample_fs = 5e9;
max_signal_per_frame = 1;
resolution = [512,512];
window_length = resolution(2);
overlap = 256+128;
path = './data_gen';
% snr_ = linspace(-24,-12,7)
snr_ = [-15,-10,-5,0,5,10,15,20];
number = 1000;
check = true;
thread = 2;
%% gen
i = 0;
gen = keysight_signal_gen(fs,resample_fs,max_signal_per_frame,...
        resolution,window_length,overlap,char(strcat(path,'_',string(i),'/')),snr_,number,check);
gen.call();
% parfor i=1:thread
%     gen = keysight_signal_gen(fs,resample_fs,max_signal_per_frame,...
%         resolution,window_length,overlap,char(strcat(path,'_',string(i),'/')),snr_,number,check);
%     gen.call();
% end
%%

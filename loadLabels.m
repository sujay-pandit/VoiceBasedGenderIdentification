%% Import data from text file.
% Script for importing data from the following text file:
%
%    F:\SNU\Machine Learning\DataSets\voice based gender identification\voice.csv
%
% To extend the code to different selected data or a different text file,
% generate a function instead of a script.

% Auto-generated by MATLAB on 2018/03/10 12:05:01

%% Initialize variables.
filename = 'F:\SNU\Machine Learning\DataSets\voice based gender identification\voice.csv';
delimiter = ',';
startRow = 2;

%% Format for each line of text:
%   column21: text (%q)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%q%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to the format.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for
% unimportable data, select unimportable cells in a file and regenerate the
% script.

%% Create output variable
voice_label = [dataArray{1:end-1}];

%% Clear temporary variables
clearvars filename delimiter startRow formatSpec fileID dataArray ans;
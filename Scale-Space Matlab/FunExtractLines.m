function result = FunExtractLines(number)
a= addpath(genpath(pwd));
currentFolder = pwd
name = fullfile(currentFolder,'tests','final_test_set','input',"atest" + string(number) + ".png");
I = imread(name);
if size(I,3)==3
    I = rgb2gray(I);
end
level = graythresh(I);
BW = imbinarize(I,level);
bin = ~BW;
[result,~, ~, ~] = ExtractLines(I, bin);% Extract the lines, linesMask = intermediate line results for debugging.
%val = label2rgb(result);

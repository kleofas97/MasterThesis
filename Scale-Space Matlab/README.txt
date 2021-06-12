==============================================================================================
Using Scale-Space Anisotropic Smoothing for Text Line Extraction in Historical Documents
==============================================================================================

1. Main functions:
------------------
This package contains the following functions
ExtractLines.m 			 - Implementation of the line extraction algorithm (Section 4)
BatchLinesScriptXXX.m 	 - A script for running the algorithm on the XXX(ICDAR/PARZIVAL/SAINT GALL) dataset images.

2. Usage example:
-----------------
Here is a small usage example. You may copy-paste these commands into Matlab:

I = imread('101.tif');
bin = ~I;		  													% ICDAR is composed of binary images. We assume that the text is brigher than the background.	
[result,Labels, linesMask, newLines] = ExtractLines(I, bin);		% Extract the lines, linesMask = intermediate line results for debugging.
imshow(label2rgb(result))											% Display result


The code for multi-skew lines is run using the following commands:

I = imread('ms_25.png');
bin = ~I;
[ result,Labels, finalLines, newLines, oldLines ] = multiSkewLinesExtraction(I, bin);
imshow(label2rgb(result))

3. Environment:
---------------
The code was tested on MATLAB 2013 using windows 8.1 and windows 7.

4. Proper reference:
-------------------
Using this software in any academic work you must cite the following work in any resulting publication:
Rafi Cohen, Itshak Dinstein, Jihad El-Sana, Klara Kedem. "Using Scale-Space Anisotropic Smoothing for Text Line Extraction in Historical Documents", ICIAR 2014

------------------------------------------

B. Resources:
-------------
1. ICDAR 2013  		 	- http://users.iit.demokritos.gr/~nstam/ICDAR2013HandSegmCont/resources.html
2. Parzival/Saint Gall  - http://www.iam.unibe.ch/fki/databases/iam-historical-document-database
3. BGU_LINE_EXTRACTION	- http://www.cs.bgu.ac.il/~abedas/dataset/Dataset_BGU_LINE_EXTRACTION.zip	


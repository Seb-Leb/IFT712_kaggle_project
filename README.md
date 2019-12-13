﻿# IFT712_kaggle_project - Fall 2019
# Jérémie Béliveau-Lefebvre 04494470
# Sebastien Leblanc 18206273
# Team repo for Usherbrooke IFT712 term project - Fall 2019


The present document has an objective of guidance for the reader on the design 
and structure of data used for that project.

The main objective of the present project is to 
 - classifiate a database from kaggle with six(6) differents classification method and 
   provide a proper analysis of these method.

Sides objecttives will be described as follow:
 - Is each method well cross-valiaded?
 - Is the researsh for hper-parameters well done?
 - Are methods trained and tested on the same data?
 - Is it possible to verify all of this in the report?
 - Have we used raw data or did we reorganised them to get better result?

Analysis of result : 
 - good or not, we have to do a proper analysis of the result. 


Main :
 - initial parameter reading
 - data gathering # function call
         - training data
         - testing data
 - for each method (show progress while doing so) 
	# 6 functions that returns results all stored in an array of 6 items
         - train
         - test
         - analyse
         - store analysis

 - show analysis results of the array (6 methods) with a cross-validation



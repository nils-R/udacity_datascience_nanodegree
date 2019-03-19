### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Description](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python versions 3.*.

## Project Motivation<a name="motivation"></a>

This blog post's aim is to explore food consumption patterns in European Countries. Specifically, it will look to answer the following questions:

1. Have there been any changes in Swedish consumption patterns between 1997-98 and 2010-2011?
2. What food categories are positively and negatively correlated with each other?
3. What countries have the greatest similarities in food patterns and what food items are representative of each cluster?

## File Descriptions <a name="files"></a>

chronicgdaytotpop.xlsx contains food consumption statistics. This dataset can be downloaded from the [efsa website](https://data.europa.eu/euodp/data/dataset/the-efsa-comprehensive-european-food-consumption-database/resource/0f73e423-b95a-408b-8e5b-a15de4fc97cf). The data is available at 4 different levels of detail, with increasing granularity of each food item. For this analysis, we will only work with food items at the L1 (e.g. "Grains and grain-based products") and L2 (e.g. "Pasta") classification levels.

There is also a notebook available. This notebook contains the code that was used to process, plot and model the data.

## Results<a name="results"></a>

The main findings of the code can be found at the [post](githubblog), as well as in the Blogpost - content notebook.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Credit to efsa for sharing the comprehensive dataset freely. 
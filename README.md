# Weather Prediction using RNN

## Introduction:
This Jupyter Notebook implements a Recurrent Neural Network (RNN) model to predict maximum temperatures for the subsequent day in Seattle, Washington, USA, based on historical weather data spanning from 2012 to 2015. The project aims to demonstrate the effectiveness of machine learning, particularly RNNs, in weather forecasting tasks.

## Dataset:
The dataset used in this notebook contains daily meteorological observations, including variables such as date, precipitation, maximum and minimum temperatures, wind speed, and weather type. The dataset provides a comprehensive overview of climatic trends and variations in Seattle over the specified four-year period.

## Approach:
1. **Data Preprocessing**: 
   - Adding a target variable column representing the maximum temperature forecasted for the following day.
   - Normalizing the data and splitting it into training and testing subsets.
   - Feature selection and extraction.

2. **Model Implementation**:
   - Configuring the RNN architecture with input, hidden, and output layers.
   - Utilizing the tanh activation function for hidden state computation.
   - Training the model and evaluating its performance using various metrics.

3. **Results and Discussion**:
   - Analysis of performance metrics including Mean Squared Error (MSE), Mean Absolute Error (MAE), R-squared Fit, and Loss.
   - Visualization of model predictions against actual values.
   
## Conclusion:
The notebook demonstrates the capability of RNNs in accurately forecasting maximum temperatures, showcasing their relevance in weather prediction tasks for informed decision-making across various sectors.

## Dependencies:
- Python 3
- Jupyter Notebook
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

## Instructions:
1. Ensure Python 3 is installed on your system.
2. Install Jupyter Notebook using `pip install jupyter`.
3. Install required libraries using `pip install pandas numpy scikit-learn matplotlib seaborn`.
4. Download the dataset and the notebook file.
5. Open the notebook in Jupyter Notebook and execute each cell sequentially to reproduce the results.

## References:
- [Link to Dataset](https://github.com/deeppadmani/Datasets/blob/main/weather/seattle-weather.csv)
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Documentation](https://seaborn.pydata.org/tutorial.html)



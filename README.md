# Ames City Housing Price Predictor
By: Darrell Amio

## Overview
This repository contains a project for predicting housing prices in Ames City, Iowa using machine learning techniques. The project explores various deployment methods, including Heroku, Docker, and AWS SageMaker, highlighting different stages and challenges encountered in deploying a machine learning model across multiple platforms.

## Problem Statement
Finding a home is like navigating a jungle of open houses, bidding wars, and endless decisions. It's a wild ride! Enter our trusty machine learning model, here to tame the beast of predicting house prices in Ames City, Iowa. By analyzing key features like location, size, and the number of bedrooms, our model can cut through the chaos and provide accurate predictions based on historical data. This isn’t just a fancy tech trick; it’s a game-changer for home buyers in Ames City. With predicted prices at their fingertips, buyers can make informed decisions, avoid overpaying, and ultimately find their dream home without losing their sanity.

## Why Machine Learning?
Machine learning is the ideal choice for predicting house prices because it excels at handling a large number of features and capturing non-linear relationships. Unlike traditional statistical methods, which can struggle with the complex interactions between variables, machine learning models can learn from these intricate patterns. Additionally, they improve over time as more data becomes available, making them increasingly accurate and reliable.

## Data Source
The project utilizes the Ames Housing dataset from Kaggle, comprising various features such as bedrooms, bathrooms, square footage, and location.
- [Data Source](https://www.kaggle.com/datasets/lespin/house-prices-dataset/data)

## Background Info/Outside Research
The real estate market in Ames City, Iowa, while not as volatile as major metropolitan areas, still presents its own set of challenges. Factors such as local economic conditions, demographic changes, and seasonal trends all play a role in influencing house prices. Understanding these dynamics is crucial for developing a reliable prediction model. Recent trends indicate a steady demand for housing in Ames, driven by its reputation as a family-friendly community with excellent schools and a high quality of life.

## References
- [Ames City Housing Market Trends](https://example.com/ames-city-housing-market)
- [Investopedia - Top U.S. Housing Market Indicators](https://www.investopedia.com/articles/personal-finance/033015/top-us-housing-market-indicators.asp)

## Projects
1. **Local Deployment and Heroku**: A machine learning model deployed on Heroku for predicting housing prices in Ames City, Iowa.
   - [Heroku App](https://your-heroku-app-link)
2. **Docker Deployment**: The application is containerized and available for public access on Docker Hub.
   - [Docker Hub](https://hub.docker.com/r/your-docker-hub-repo)
3. **AWS Deployment**: Initial setup on AWS includes the use of S3 for storage, ECR for Docker image management, and a SageMaker instance for model development. Full deployment on AWS is ongoing and aims for a robust, scalable solution.

## Repository Structure
- `/Part1_LocalDeployment`: Files for local deployment and Heroku, including Docker configuration.
- `/Part2_AWS_Deployment`: Contains documentation and screenshots illustrating the setup of AWS resources, with further deployment under development.

## Usage
Detailed instructions for running and deploying the models are provided in each subdirectory.

## Additional Resources
- **Ames Housing Price Predictor.pdf**: Contains a PowerPoint overview of the project, detailing each phase from development to deployment and outlining the challenges encountered.

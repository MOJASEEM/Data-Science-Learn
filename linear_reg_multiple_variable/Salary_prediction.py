import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from word2number import w2n as word2number  

# Load the dataset
df = pd.read_csv('d:/Data Science/linear_reg_multiple_variable/hiring.csv')
df['experience'] = df['experience'].fillna('zero')
# Convert 'experience' column from words to numbers
df.experience = df.experience.apply(word2number.word_to_num)
# Fill missing values in 'experience' column with the average experience
average_experience = math.floor((df['experience']).median())
experience = df.experience.replace(0,average_experience)
# Fill missing values in 'test_score' column with the average test score
df['test_score'] = df['test_score'].fillna(0)
average_test_score = math.floor(df['test_score'].median())
test_score = df.test_score.replace(0,average_test_score)
# Prepare the feature matrix and target vector for training
reg = LinearRegression()
reg.fit(df[['experience', 'test_score', 'interview_score']], df.salary)
data=pd.read_csv('d:/Data Science/linear_reg_multiple_variable/interview_score.csv')
data['experience'] = data['experience'].fillna('zero')
# Convert 'experience' column from words to numbers
data.experience = data.experience.apply(word2number.word_to_num)
# Fill missing values in 'experience' column with the average experience
average_experience = math.floor((data['experience']).median())
experience = data.experience.replace(0,average_experience)
# Predict salaries for new employees
predicted_salaries = reg.predict(data)
data['predicted_salary'] = predicted_salaries
data.to_csv('D:/Data Science/linear_reg_multiple_variable/expected_salaries.csv', index=False)   
# Predict the salary for a candidate with 2 years of experience, a test score of 9.0, and an interview score of 6.0
x=int(input("Enter experience in years: "))
y=float(input("Enter test score out of 10: "))
z=float(input("Enter interview score out of 10: "))
sample = pd.DataFrame([[x,y,z]], columns=['experience', 'test_score', 'interview_score'])
predicted_salary = reg.predict(sample)
print(predicted_salary)
print(f"The predicted salary for a candidate with {x} years of experience, a test score of {y}, and an interview score of {z} is: {predicted_salary[0]}")

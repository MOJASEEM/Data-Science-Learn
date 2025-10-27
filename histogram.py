import matplotlib.pyplot as plt

blood_sugar_men = [90, 85, 88, 92, 95, 100, 110, 105, 98, 102]
blood_sugar_women = [80, 82, 78, 85, 88, 90, 95, 92, 87, 89]
plt.xlabel('Blood Sugar Level (mg/dL)')
plt.ylabel('Number of Individuals')
plt.title('Blood Sugar Level Distribution by Gender')
plt.hist([blood_sugar_men,blood_sugar_women], bins=3, alpha=0.5, label=['Men','Women'], color=['Green','Red'])
plt.legend()   
plt.show()
blood_sugar_men_avg = sum(blood_sugar_men) / len(blood_sugar_men)
blood_sugar_women_avg = sum(blood_sugar_women) / len(blood_sugar_women)
print(f'Average Blood Sugar Level for men: {blood_sugar_men_avg:.2f} mg/dL')
print(f'Average Blood Sugar Level for women: {blood_sugar_women_avg:.2f} mg/dL')   
# Blood sugar levels are measured in mg/dL (milligrams per deciliter)   
# This script visualizes the distribution of blood sugar levels
import matplotlib.pyplot as plt
blood_sugar_men = [90, 110, 85, 95, 120, 130, 100, 115, 105, 125] 
blood_sugar_women = [80, 95, 100, 85, 90, 110, 105, 115, 120, 130]  
plt.boxplot([blood_sugar_men, blood_sugar_women], labels    =['Men', 'women'])
plt.title('Blood Sugar Levels by Gender')
plt.ylabel('Blood Sugar Level (mg/dL)')
plt.grid(True)
plt.show()
# Blood sugar levels for men and women  
        

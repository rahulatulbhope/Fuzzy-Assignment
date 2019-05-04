import numpy as np
import pandas as pd
dataset1 = pd.read_csv("fuzzy_set_1.csv")
dataset2 = pd.read_csv("fuzzy_set_2.csv")
dataset3 = pd.read_csv("fuzzy_set_3.csv")


X_1 = dataset1.iloc[:, 0:4].values
y_1 = dataset1.iloc[:, 4].values

X_2 = dataset2.iloc[:, 0:4].values
y_2 = dataset2.iloc[:, 4].values

X_3 = dataset3.iloc[:, 0:4].values
y_3 = dataset3.iloc[:, 4].values


#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=1)
X_final_1 = lda.fit_transform(X_1, y_1)
X_final_2 = lda.fit_transform(X_2, y_2)
X_final_3 = lda.fit_transform(X_3, y_3)

#X_new = np.concatenate(X_final_1,X_final_2,X_final_3)

#print(X_new)

np.savetxt("model_1_new.csv", X_final_1, delimiter=",")
np.savetxt("model_2_new.csv", X_final_2, delimiter=",")
np.savetxt("model_3_new.csv", X_final_3, delimiter=",")


'''
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# New Antecedent/Consequent objects hold universe variables and membership
# functions


#BFJ

x = ctrl.Antecedent(np.arange(0,1,0.1), 'x')
y = ctrl.Antecedent(np.arange(0,1,0.1), 'y')
z = ctrl.Antecedent(np.arange(0,1,0.1), 'z')

epilepsy = ctrl.Consequent(np.arange(0, 1, 0.1), 'epilepsy')

# Auto-membership function population is possible with .automf(3, 5, or 7)

names = ['low', 'medium', 'high']

x.automf(3)
y.automf(3)
z.automf(3)



# Custom membership functions can be built interactively with a familiar,
# Pythonic API
epilepsy['low'] = fuzz.trimf(epilepsy.universe, [0, 0.25, 0.5])
epilepsy['medium'] = fuzz.trimf(epilepsy.universe, [0.25, 0.5, 0.75])
epilepsy['high'] = fuzz.trimf(epilepsy.universe, [0.5, 0.75, 1])

# You can see how these look with .view()

epilepsy.view()
plt.show()


rule1 = ctrl.Rule(x['average'] & y['average'] & z['good'], epilepsy['low'])
rule2 = ctrl.Rule(x['average'] & y['average'] & z['good'], epilepsy['medium'])
rule3 = ctrl.Rule(x['good'] & y['average'] & z['good'], epilepsy['high'])
#rule1.view()

epilepsyping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

epilepsyping = ctrl.ControlSystemSimulation(epilepsyping_ctrl)

# Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
# Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
epilepsyping.input['x'] = X_final_1[1]
epilepsyping.input['y'] = X_final_2[1]
epilepsyping.input['z'] = X_final_3[1]

# Crunch the numbers
epilepsyping.compute()

print(epilepsyping.output['epilepsy'])
epilepsy.view(sim=epilepsyping)
'''